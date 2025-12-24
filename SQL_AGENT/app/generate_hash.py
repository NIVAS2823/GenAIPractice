import os

fake_users_db = {
    os.getenv("ADMIN_USERNAME", "admin"): {
        "username": os.getenv("ADMIN_USERNAME", "admin"),
        "hashed_password": os.getenv("ADMIN_PASSWORD_HASH"),
    }
}
