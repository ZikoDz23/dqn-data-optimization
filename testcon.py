import psycopg2
from psycopg2 import OperationalError

def check_connection():
    try:
        # Connect to the PostgreSQL database
        connection = psycopg2.connect(
            host="localhost",
            database="imdbload",   
            user="postgres",
            password="123456789"
        )
        
        # Create a cursor object to interact with the database
        cursor = connection.cursor()
        
        # Run a simple query to check the connection
        cursor.execute("SELECT version();")
        
        # Fetch and print the result
        result = cursor.fetchone()
        print("Connection successful:", result)

        # Close the cursor and connection
        cursor.close()
        connection.close()

    except OperationalError as e:
        print("Error:", e)

# Call the function
check_connection()
