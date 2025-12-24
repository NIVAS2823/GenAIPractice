from app.auth import hash_password
import os 

_admin_username = os.getenv("ADMIN_USERNAME")
_admin_password = os.getenv("ADMIN_PASSWORD")

if not _admin_username and _admin_password:
    raise RuntimeError("Admin credentials invalid")



fake_users_db = {
    "admin":{
        "username":_admin_username,
        "hashed_password":hash_password(_admin_password)
    }
}