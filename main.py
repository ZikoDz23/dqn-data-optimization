from src.train import train

db_config = {
    "host": "localhost",
    "database": "imdbload",
    "user": "postgres",
    "password": "123456789"
}

if __name__ == "__main__":
    train(db_config)
