"""
Database models for ChemAlize application.
"""
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Integer, String, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import json

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """User model - supports both anonymous and registered users."""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String(36), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=True, index=True)
    username = db.Column(db.String(80), unique=True, nullable=True)
    password_hash = db.Column(db.String(255), nullable=True)
    is_anonymous_user = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_activity = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)
    data_quota_mb = db.Column(db.Integer, default=100)

    # Relationships
    datasets = db.relationship('Dataset', backref='owner', lazy='dynamic', cascade='all, delete-orphan')
    analysis_results = db.relationship('AnalysisResult', backref='user', lazy='dynamic', cascade='all, delete-orphan')

    def set_password(self, password):
        """Hash and set password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Check if password matches hash."""
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()
        db.session.commit()

    def get_current_dataset(self):
        """Get user's current active dataset."""
        return Dataset.query.filter_by(user_id=self.id, is_current=True).first()

    def get_disk_usage_mb(self):
        """Calculate total disk usage in MB."""
        total_kb = db.session.query(db.func.sum(Dataset.file_size_kb))\
            .filter(Dataset.user_id == self.id)\
            .scalar() or 0
        return total_kb / 1024

    def __repr__(self):
        return f'<User {self.uuid} ({self.email or "anonymous"})>'


class Dataset(db.Model):
    """Dataset model - uploaded data files."""
    __tablename__ = 'datasets'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    filename = db.Column(db.String(255), nullable=False)  # Original filename
    stored_filename = db.Column(db.String(255), nullable=False)  # UUID_filename
    file_type = db.Column(db.String(10), nullable=False)
    file_size_kb = db.Column(db.Integer)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    last_modified = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = db.Column(db.String(20), default='uploaded')  # uploaded, cleaned, processing, error
    num_rows = db.Column(db.Integer)
    num_columns = db.Column(db.Integer)
    column_names = db.Column(db.Text)  # JSON array
    is_current = db.Column(db.Boolean, default=False, index=True)

    # Relationships
    analysis_results = db.relationship('AnalysisResult', backref='dataset', lazy='dynamic', cascade='all, delete-orphan')
    preprocessing_history = db.relationship('PreprocessingHistory', backref='dataset', lazy='dynamic', cascade='all, delete-orphan')

    def set_column_names(self, columns):
        """Set column names as JSON."""
        self.column_names = json.dumps(columns)

    def get_column_names(self):
        """Get column names from JSON."""
        if self.column_names:
            return json.loads(self.column_names)
        return []

    def make_current(self):
        """Make this dataset the current one for the user."""
        # Unset all other datasets
        Dataset.query.filter_by(user_id=self.user_id, is_current=True).update({'is_current': False})
        self.is_current = True
        db.session.commit()

    def __repr__(self):
        return f'<Dataset {self.filename} (user={self.user_id})>'


class AnalysisResult(db.Model):
    """Analysis result model - stores ML analysis results."""
    __tablename__ = 'analysis_results'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id', ondelete='CASCADE'), nullable=False, index=True)
    analysis_type = db.Column(db.String(50), nullable=False, index=True)
    parameters = db.Column(db.Text)  # JSON
    results = db.Column(db.Text)  # JSON
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    execution_time_ms = db.Column(db.Integer)
    status = db.Column(db.String(20), default='completed')  # completed, failed, running
    error_message = db.Column(db.Text)

    # Relationships
    plots = db.relationship('AnalysisPlot', backref='analysis', lazy='dynamic', cascade='all, delete-orphan')

    def set_parameters(self, params_dict):
        """Set parameters as JSON."""
        self.parameters = json.dumps(params_dict)

    def get_parameters(self):
        """Get parameters from JSON."""
        if self.parameters:
            return json.loads(self.parameters)
        return {}

    def set_results(self, results_dict):
        """Set results as JSON."""
        self.results = json.dumps(results_dict)

    def get_results(self):
        """Get results from JSON."""
        if self.results:
            return json.loads(self.results)
        return {}

    def __repr__(self):
        return f'<AnalysisResult {self.analysis_type} (dataset={self.dataset_id})>'


class AnalysisPlot(db.Model):
    """Analysis plot model - stores plot files from analyses."""
    __tablename__ = 'analysis_plots'

    id = db.Column(db.Integer, primary_key=True)
    analysis_result_id = db.Column(db.Integer, db.ForeignKey('analysis_results.id', ondelete='CASCADE'), nullable=False, index=True)
    plot_type = db.Column(db.String(50), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(512), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<AnalysisPlot {self.plot_type} ({self.filename})>'


class PreprocessingHistory(db.Model):
    """Preprocessing history - tracks data transformations."""
    __tablename__ = 'preprocessing_history'

    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id', ondelete='CASCADE'), nullable=False, index=True)
    operation = db.Column(db.String(50), nullable=False)
    parameters = db.Column(db.Text)  # JSON
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def set_parameters(self, params_dict):
        """Set parameters as JSON."""
        self.parameters = json.dumps(params_dict)

    def get_parameters(self):
        """Get parameters from JSON."""
        if self.parameters:
            return json.loads(self.parameters)
        return {}

    def __repr__(self):
        return f'<PreprocessingHistory {self.operation} (dataset={self.dataset_id})>'


# Helper functions
def get_or_create_anonymous_user(uuid):
    """Get or create an anonymous user by UUID."""
    user = User.query.filter_by(uuid=uuid).first()
    if not user:
        user = User(uuid=uuid, is_anonymous_user=True)
        db.session.add(user)
        db.session.commit()
    else:
        user.update_activity()
    return user


def cleanup_old_anonymous_users(days=30):
    """Remove anonymous users inactive for more than specified days."""
    from datetime import timedelta
    cutoff = datetime.utcnow() - timedelta(days=days)

    old_users = User.query\
        .filter(User.is_anonymous_user == True)\
        .filter(User.last_activity < cutoff)\
        .all()

    count = len(old_users)
    for user in old_users:
        db.session.delete(user)  # CASCADE will delete all related data

    db.session.commit()
    return count


def get_database_stats():
    """Get database statistics."""
    return {
        'total_users': User.query.count(),
        'anonymous_users': User.query.filter_by(is_anonymous_user=True).count(),
        'registered_users': User.query.filter_by(is_anonymous_user=False).count(),
        'total_datasets': Dataset.query.count(),
        'total_analyses': AnalysisResult.query.count(),
        'total_disk_kb': db.session.query(db.func.sum(Dataset.file_size_kb)).scalar() or 0
    }
