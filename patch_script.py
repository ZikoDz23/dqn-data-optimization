import sys

TARGET = "rl_query_optimizer/imdbpy2sql.py"

with open(TARGET, "r") as f:
    content = f.read()

PATCH = """import pkgutil
import importlib.util

if not hasattr(pkgutil, 'find_loader'):
    def find_loader(fullname):
        spec = importlib.util.find_spec(fullname)
        return spec.loader if spec else None
    pkgutil.find_loader = find_loader

from imdb.parser.sql.dbschema import DB_SCHEMA, dropTables, createTables, createIndexes
from imdb.parser.sql import soundex
from imdb.utils import analyze_title, analyze_name, date_and_notes, \\
    build_name, build_title, normalizeName, normalizeTitle, _articles, \\
        build_company_name, analyze_company_name, canonicalTitle
from imdb._exceptions import IMDbParserError, IMDbError
from imdb.parser.sql.alchemyadapter import getDBTables, setConnection
import imdb.parser.sql.alchemyadapter
from sqlalchemy import create_engine as _real_create_engine
from sqlalchemy import Index, exc

def _patched_create_engine(*args, **kwargs):
    kwargs.pop('encoding', None)
    return _real_create_engine(*args, **kwargs)

imdb.parser.sql.alchemyadapter.create_engine = _patched_create_engine

def _patched_dropTable(self, checkfirst=True):
    from imdb.parser.sql.alchemyadapter import metadata
    self.table.drop(bind=metadata.bind, checkfirst=checkfirst)

imdb.parser.sql.alchemyadapter.TableAdapter.dropTable = _patched_dropTable

def _patched_createTable(self, checkfirst=True):
    from imdb.parser.sql.alchemyadapter import metadata
    self.table.create(bind=metadata.bind, checkfirst=checkfirst)
    for col in self._imdbpySchema.cols:
        if col.name == 'id':
            continue
        if col.params.get('alternateID', False):
            self._createIndex(col, checkfirst=checkfirst)

imdb.parser.sql.alchemyadapter.TableAdapter.createTable = _patched_createTable

def _patched_createIndex(self, col, checkfirst=True):
    from imdb.parser.sql.alchemyadapter import metadata
    from imdb.parser.sql.alchemyadapter import UNICODECOL, STRINGCOL
    
    idx_name = '%s_%s' % (self.table.name, col.index or col.name)
    if checkfirst:
        for index in self.table.indexes:
            if index.name == idx_name:
                return
    index_args = {}
    if self.connectionURI.startswith('mysql'):
        if col.indexLen:
            index_args['mysql_length'] = col.indexLen
        elif col.kind in (UNICODECOL, STRINGCOL):
            index_args['mysql_length'] = min(5, col.params.get('length') or 5)
    
    idx = Index(idx_name, getattr(self.table.c, self.colMap[col.name]), **index_args)
    try:
        idx.create(bind=metadata.bind)
    except exc.OperationalError as e:
        pass

imdb.parser.sql.alchemyadapter.TableAdapter._createIndex = _patched_createIndex

def _patched_call(self, *args, **kwds):
    from imdb.parser.sql.alchemyadapter import metadata
    taArgs = {}
    for key, value in list(kwds.items()):
        taArgs[self.colMap.get(key, key)] = value
    metadata.bind.execute(self._ta_insert, taArgs)

imdb.parser.sql.alchemyadapter.TableAdapter.__call__ = _patched_call

def _patched_select(self, conditions=None):
    from imdb.parser.sql.alchemyadapter import metadata, ResultAdapter
    from sqlalchemy import select
    stmt = select(self.table)
    if conditions is not None:
        stmt = stmt.where(conditions)
    result = metadata.bind.execute(stmt)
    return ResultAdapter(result, self.table, colMap=self.colMap)

imdb.parser.sql.alchemyadapter.TableAdapter.select = _patched_select

def _patched_setConnection(uri, tables, encoding='utf8', debug=False):
    params = {'encoding': encoding}
    if uri.startswith('mysql'):
        if '?' in uri: uri += '&'
        else: uri += '?'
        uri += 'charset=%s' % encoding
    if debug: params['echo'] = True
    if uri.startswith('ibm_db'): params['convert_unicode'] = True
    
    engine = imdb.parser.sql.alchemyadapter.create_engine(uri, **params)
    eng_conn = engine.connect()
    
    imdb.parser.sql.alchemyadapter.metadata.bind = eng_conn
    
    if uri.startswith('sqlite'):
        major = sys.version_info[0]
        minor = sys.version_info[1]
        if major > 2 or (major == 2 and minor > 5):
            eng_conn.connection.connection.text_factory = str

    class SAConnectionProxy:
        def __init__(self, sa_conn):
            self.sa_conn = sa_conn
            self.module = sa_conn.dialect.dbapi
            self.paramstyle = sa_conn.dialect.paramstyle
            self.dbName = engine.url.drivername
        def getConnection(self):
            return self
        def cursor(self):
            return self.sa_conn.connection.cursor()
        def commit(self):
            self.sa_conn.commit()
        def rollback(self):
            print("DEBUG: Rolling back transaction...")
            try:
                self.sa_conn.connection.rollback()
            except Exception:
                pass
            self.sa_conn.rollback()
        def close(self):
            self.sa_conn.close()
        def __getattr__(self, name):
            return getattr(self.sa_conn, name)

    return SAConnectionProxy(eng_conn)

imdb.parser.sql.alchemyadapter.setConnection = _patched_setConnection
"""

import re
old_imports = \"\"\"from imdb.parser.sql.dbschema import DB_SCHEMA, dropTables, createTables, createIndexes
from imdb.parser.sql import soundex
from imdb.utils import analyze_title, analyze_name, date_and_notes, \\
    build_name, build_title, normalizeName, normalizeTitle, _articles, \\
        build_company_name, analyze_company_name, canonicalTitle
from imdb._exceptions import IMDbParserError, IMDbError
from imdb.parser.sql.alchemyadapter import getDBTables, setConnection\"\"\"

content = content.replace("conn = setConnection(URI, DB_TABLES)", "conn = imdb.parser.sql.alchemyadapter.setConnection(URI, DB_TABLES)")
content = content.replace(old_imports, PATCH)

new_kind_logic = \"\"\"            if kind not in KIND_IDS:
                print('WARNING: kind "%s" not in database; adding it.' % kind)
                try:
                    # Fallback to direct SQL because ORM wrapper might be failing
                    tbl = tableName(KindType)
                    CURS.execute("INSERT INTO %s (kind) VALUES (%%s)" % tbl, (kind,))
                    connectObject.commit()
                    CURS.execute("SELECT id FROM %s WHERE kind = %%s" % tbl, (kind,))
                    res = CURS.fetchone()
                    if res:
                        KIND_IDS[kind] = res[0]
                    else:
                         print("ERROR: Inserted kind but could not find ID")
                except Exception as e:
                    print('ERROR adding kind: %s' % e)
                    try: connectObject.rollback()
                    except: pass
            if kind == 'episode':\"\"\"

content = content.replace("            if kind == 'episode':", new_kind_logic)


new_store_logic = \"\"\"    if _get_imdbids_method() == 'table':
        try:
            try:
                CURS.execute('DROP TABLE %s_extract' % table_name)
            except:
                print("DEBUG: Drop failed, rolling back")
                connectObject.rollback()
            try:
                CURS.execute('SELECT * FROM %s LIMIT 1' % table_name)
            except Exception as e:
                print("DEBUG: Select failed (%s), rolling back" % str(e))
                connectObject.rollback()
                print('missing "%s" table (ok if this is the first run)' % table_name)
                return
            query = 'CREATE TEMPORARY TABLE %s_extract AS SELECT %s, %s FROM %s WHERE %s IS NOT NULL' % \\
                    (table_name, md5sum_col, imdbID_col,
                     table_name, imdbID_col)
            CURS.execute(query)
            CURS.execute('CREATE INDEX %s_md5sum_idx ON %s_extract (%s)' % (table_name, table_name, md5sum_col))
            CURS.execute('CREATE INDEX %s_imdbid_idx ON %s_extract (%s)' % (table_name, table_name, imdbID_col))
            rows = _countRows('%s_extract' % table_name)
            print('DONE! (%d entries using a temporary table)' % rows)
            return
        except Exception as e:
            print("DEBUG: Generic error (%s), rolling back" % str(e))
            connectObject.rollback()
            print('WARNING: unable to store imdbIDs in a temporary table (falling back to dbm): %s' % e)
    try:
        db = dbm.open(_imdbIDsFileName('%s_imdbIDs.db' % cname), 'c')
    except Exception as e:
        print('WARNING: unable to store imdbIDs: %s' % str(e))
        return
    try:
        CURS.execute('SELECT %s, %s FROM %s WHERE %s IS NOT NULL' %
                     (md5sum_col, imdbID_col, table_name, imdbID_col))
        res = CURS.fetchmany(10000)
        while res:
            db.update(dict((str(x[0]), str(x[1])) for x in res))
            res = CURS.fetchmany(10000)
    except Exception as e:
        print("DEBUG: Fetch error (%s), rolling back" % str(e))
        connectObject.rollback()
        print('SKIPPING: unable to retrieve data: %s' % e)
        return\"\"\"

old_store_logic = \"\"\"    if _get_imdbids_method() == 'table':
        try:
            try:
                CURS.execute('DROP TABLE %s_extract' % table_name)
            except:
                pass
            try:
                CURS.execute('SELECT * FROM %s LIMIT 1' % table_name)
            except Exception as e:
                print('missing "%s" table (ok if this is the first run)' % table_name)
                return
            query = 'CREATE TEMPORARY TABLE %s_extract AS SELECT %s, %s FROM %s WHERE %s IS NOT NULL' % \\
                    (table_name, md5sum_col, imdbID_col,
                     table_name, imdbID_col)
            CURS.execute(query)
            CURS.execute('CREATE INDEX %s_md5sum_idx ON %s_extract (%s)' % (table_name, table_name, md5sum_col))
            CURS.execute('CREATE INDEX %s_imdbid_idx ON %s_extract (%s)' % (table_name, table_name, imdbID_col))
            rows = _countRows('%s_extract' % table_name)
            print('DONE! (%d entries using a temporary table)' % rows)
            return
        except Exception as e:
            print('WARNING: unable to store imdbIDs in a temporary table (falling back to dbm): %s' % e)
    try:
        db = dbm.open(_imdbIDsFileName('%s_imdbIDs.db' % cname), 'c')
    except Exception as e:
        print('WARNING: unable to store imdbIDs: %s' % str(e))
        return
    try:
        CURS.execute('SELECT %s, %s FROM %s WHERE %s IS NOT NULL' %
                     (md5sum_col, imdbID_col, table_name, imdbID_col))
        res = CURS.fetchmany(10000)
        while res:
            db.update(dict((str(x[0]), str(x[1])) for x in res))
            res = CURS.fetchmany(10000)
    except Exception as e:
        print('SKIPPING: unable to retrieve data: %s' % e)
        return\"\"\"

content = content.replace(old_store_logic, new_store_logic)

with open(TARGET, "w") as f:
    f.write(content)

