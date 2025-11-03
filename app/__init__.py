from flask import Flask, session, g
from flask_session import Session
from flask_login import LoginManager
from flask_migrate import Migrate
import os
from app.config import SESSION_CONFIG, ensure_directories, get_user_id

# Initialize extensions
login_manager = LoginManager()

app = Flask(__name__)

# --- KONFIGURACJA ---
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "HelloWorld")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///chemalize.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Large file upload configuration (important for Cloudflare Tunnel)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH", 524288000))  # 500MB default
app.config["UPLOAD_CHUNK_SIZE"] = int(os.getenv("CHUNK_SIZE", 5242880))  # 5MB chunks

app.config.update(SESSION_CONFIG)

# Initialize database
from app.models import db, get_or_create_anonymous_user
db.init_app(app)

# Initialize migrations
migrate = Migrate(app, db)

# Initialize login manager
login_manager.init_app(app)
login_manager.login_view = 'auth.login'  # Redirect to login if not authenticated

@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login."""
    from app.models import User
    return User.query.get(int(user_id))

# Ensure global directories exist
ensure_directories()

Session(app)  # aktywacja Flask-Session

# Middleware to initialize user session and directories
@app.before_request
def initialize_user_session():
    """Initialize user_id and create user directories before each request."""
    from flask_login import current_user
    from app.models import User

    # Get or create user in database
    if current_user.is_authenticated:
        # Logged in user
        g.user = current_user
        g.user.update_activity()
    else:
        # Anonymous user - use UUID from session
        user_uuid = get_user_id()
        g.user = get_or_create_anonymous_user(user_uuid)

    # Create user-specific directories
    ensure_directories(g.user.uuid)

# Create tables
with app.app_context():
    db.create_all()

from app import routes

