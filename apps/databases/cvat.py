from apps.databases import db



class AccountEmailaddres(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'account_emailaddress'

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    email = db.Column(db.String(254), nullable=False, unique=True)
    verified = db.Column(db.Boolean, nullable=False)
    primary = db.Column(db.Boolean, nullable=False)
    user_id = db.Column(db.ForeignKey('auth_user.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)

    user = db.relationship('AuthUser', primaryjoin='AccountEmailaddres.user_id == AuthUser.id', backref='account_emailaddress')



class AccountEmailconfirmation(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'account_emailconfirmation'

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    created = db.Column(db.DateTime(True), nullable=False)
    sent = db.Column(db.DateTime(True))
    key = db.Column(db.String(64), nullable=False, unique=True)
    email_address_id = db.Column(db.ForeignKey('account_emailaddress.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)

    email_address = db.relationship('AccountEmailaddres', primaryjoin='AccountEmailconfirmation.email_address_id == AccountEmailaddres.id', backref='account_emailconfirmations')



class AuthGroup(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'auth_group'

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    name = db.Column(db.String(150), nullable=False, unique=True)



class AuthGroupPermission(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'auth_group_permissions'
    __table_args__ = (
        db.UniqueConstraint('group_id', 'permission_id'),
    )

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    group_id = db.Column(db.ForeignKey('auth_group.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)
    permission_id = db.Column(db.ForeignKey('auth_permission.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)

    group = db.relationship('AuthGroup', primaryjoin='AuthGroupPermission.group_id == AuthGroup.id', backref='auth_group_permissions')
    permission = db.relationship('AuthPermission', primaryjoin='AuthGroupPermission.permission_id == AuthPermission.id', backref='auth_group_permissions')



class AuthPermission(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'auth_permission'
    __table_args__ = (
        db.UniqueConstraint('content_type_id', 'codename'),
    )

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    name = db.Column(db.String(255), nullable=False)
    content_type_id = db.Column(db.ForeignKey('django_content_type.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)
    codename = db.Column(db.String(100), nullable=False)

    content_type = db.relationship('DjangoContentType', primaryjoin='AuthPermission.content_type_id == DjangoContentType.id', backref='auth_permissions')



class AuthUser(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'auth_user'

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    password = db.Column(db.String(128), nullable=False)
    last_login = db.Column(db.DateTime(True))
    is_superuser = db.Column(db.Boolean, nullable=False)
    username = db.Column(db.String(150), nullable=False, unique=True)
    first_name = db.Column(db.String(150), nullable=False)
    last_name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(254), nullable=False)
    is_staff = db.Column(db.Boolean, nullable=False)
    is_active = db.Column(db.Boolean, nullable=False)
    date_joined = db.Column(db.DateTime(True), nullable=False)



class AuthUserGroup(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'auth_user_groups'
    __table_args__ = (
        db.UniqueConstraint('user_id', 'group_id'),
    )

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    user_id = db.Column(db.ForeignKey('auth_user.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)
    group_id = db.Column(db.ForeignKey('auth_group.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)

    group = db.relationship('AuthGroup', primaryjoin='AuthUserGroup.group_id == AuthGroup.id', backref='auth_user_groups')
    user = db.relationship('AuthUser', primaryjoin='AuthUserGroup.user_id == AuthUser.id', backref='auth_user_groups')



class AuthUserUserPermission(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'auth_user_user_permissions'
    __table_args__ = (
        db.UniqueConstraint('user_id', 'permission_id'),
    )

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    user_id = db.Column(db.ForeignKey('auth_user.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)
    permission_id = db.Column(db.ForeignKey('auth_permission.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)

    permission = db.relationship('AuthPermission', primaryjoin='AuthUserUserPermission.permission_id == AuthPermission.id', backref='auth_user_user_permissions')
    user = db.relationship('AuthUser', primaryjoin='AuthUserUserPermission.user_id == AuthUser.id', backref='auth_user_user_permissions')



class AuthtokenToken(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'authtoken_token'

    key = db.Column(db.String(40), primary_key=True, index=True)
    created = db.Column(db.DateTime(True), nullable=False)
    user_id = db.Column(db.ForeignKey('auth_user.id', deferrable=True, initially='DEFERRED'), nullable=False, unique=True)

    user = db.relationship('AuthUser', uselist=False, primaryjoin='AuthtokenToken.user_id == AuthUser.id', backref='authtoken_tokens')



class DjangoAdminLog(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'django_admin_log'
    __table_args__ = (
        db.CheckConstraint('action_flag >= 0'),
    )

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    action_time = db.Column(db.DateTime(True), nullable=False)
    object_id = db.Column(db.Text)
    object_repr = db.Column(db.String(200), nullable=False)
    action_flag = db.Column(db.SmallInteger, nullable=False)
    change_message = db.Column(db.Text, nullable=False)
    content_type_id = db.Column(db.ForeignKey('django_content_type.id', deferrable=True, initially='DEFERRED'), index=True)
    user_id = db.Column(db.ForeignKey('auth_user.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)

    content_type = db.relationship('DjangoContentType', primaryjoin='DjangoAdminLog.content_type_id == DjangoContentType.id', backref='django_admin_logs')
    user = db.relationship('AuthUser', primaryjoin='DjangoAdminLog.user_id == AuthUser.id', backref='django_admin_logs')



class DjangoContentType(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'django_content_type'
    __table_args__ = (
        db.UniqueConstraint('app_label', 'model'),
    )

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    app_label = db.Column(db.String(100), nullable=False)
    model = db.Column(db.String(100), nullable=False)



class DjangoMigration(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'django_migrations'

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    app = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    applied = db.Column(db.DateTime(True), nullable=False)



class DjangoSession(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'django_session'

    session_key = db.Column(db.String(40), primary_key=True, index=True)
    session_data = db.Column(db.Text, nullable=False)
    expire_date = db.Column(db.DateTime(True), nullable=False, index=True)



class DjangoSite(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'django_site'

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    domain = db.Column(db.String(100), nullable=False, unique=True)
    name = db.Column(db.String(50), nullable=False)



class EngineAttributespec(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_attributespec'
    __table_args__ = (
        db.UniqueConstraint('label_id', 'name'),
    )

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    label_id = db.Column(db.ForeignKey('engine_label.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)
    default_value = db.Column(db.String(128), nullable=False)
    input_type = db.Column(db.String(16), nullable=False)
    mutable = db.Column(db.Boolean, nullable=False)
    name = db.Column(db.String(64), nullable=False)
    values = db.Column(db.String(4096), nullable=False)

    label = db.relationship('EngineLabel', primaryjoin='EngineAttributespec.label_id == EngineLabel.id', backref='engine_attributespecs')



class EngineClientfile(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_clientfile'
    __table_args__ = (
        db.UniqueConstraint('data_id', 'file'),
    )

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    file = db.Column(db.String(1024), nullable=False)
    data_id = db.Column(db.ForeignKey('engine_data.id', deferrable=True, initially='DEFERRED'), index=True)

    data = db.relationship('EngineDatum', primaryjoin='EngineClientfile.data_id == EngineDatum.id', backref='engine_clientfiles')



class EngineCloudstorage(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_cloudstorage'
    __table_args__ = (
        db.UniqueConstraint('provider_type', 'resource', 'credentials'),
    )

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    provider_type = db.Column(db.String(20), nullable=False)
    resource = db.Column(db.String(222), nullable=False)
    display_name = db.Column(db.String(63), nullable=False)
    created_date = db.Column(db.DateTime(True), nullable=False)
    updated_date = db.Column(db.DateTime(True), nullable=False)
    credentials = db.Column(db.String(500), nullable=False)
    credentials_type = db.Column(db.String(29), nullable=False)
    specific_attributes = db.Column(db.String(1024), nullable=False)
    description = db.Column(db.Text, nullable=False)
    owner_id = db.Column(db.ForeignKey('auth_user.id', deferrable=True, initially='DEFERRED'), index=True)
    organization_id = db.Column(db.ForeignKey('organizations_organization.id', deferrable=True, initially='DEFERRED'), index=True)

    organization = db.relationship('OrganizationsOrganization', primaryjoin='EngineCloudstorage.organization_id == OrganizationsOrganization.id', backref='engine_cloudstorages')
    owner = db.relationship('AuthUser', primaryjoin='EngineCloudstorage.owner_id == AuthUser.id', backref='engine_cloudstorages')



class EngineComment(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_comment'

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    message = db.Column(db.Text, nullable=False)
    created_date = db.Column(db.DateTime(True), nullable=False)
    updated_date = db.Column(db.DateTime(True), nullable=False)
    owner_id = db.Column(db.ForeignKey('auth_user.id', deferrable=True, initially='DEFERRED'), index=True)
    issue_id = db.Column(db.ForeignKey('engine_issue.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)

    issue = db.relationship('EngineIssue', primaryjoin='EngineComment.issue_id == EngineIssue.id', backref='engine_comments')
    owner = db.relationship('AuthUser', primaryjoin='EngineComment.owner_id == AuthUser.id', backref='engine_comments')



class EngineDatum(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_data'
    __table_args__ = (
        db.CheckConstraint('chunk_size >= 0'),
        db.CheckConstraint('image_quality >= 0'),
        db.CheckConstraint('size >= 0'),
        db.CheckConstraint('start_frame >= 0'),
        db.CheckConstraint('stop_frame >= 0')
    )

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    chunk_size = db.Column(db.Integer)
    size = db.Column(db.Integer, nullable=False)
    image_quality = db.Column(db.SmallInteger, nullable=False)
    start_frame = db.Column(db.Integer, nullable=False)
    stop_frame = db.Column(db.Integer, nullable=False)
    frame_filter = db.Column(db.String(256), nullable=False)
    compressed_chunk_type = db.Column(db.String(32), nullable=False)
    original_chunk_type = db.Column(db.String(32), nullable=False)
    storage_method = db.Column(db.String(15), nullable=False)
    storage = db.Column(db.String(15), nullable=False)
    cloud_storage_id = db.Column(db.ForeignKey('engine_cloudstorage.id', deferrable=True, initially='DEFERRED'), index=True)
    sorting_method = db.Column(db.String(15), nullable=False)
    deleted_frames = db.Column(db.Text, nullable=False)

    cloud_storage = db.relationship('EngineCloudstorage', primaryjoin='EngineDatum.cloud_storage_id == EngineCloudstorage.id', backref='engine_data')



class EngineImage(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_image'
    __table_args__ = (
        db.CheckConstraint('frame >= 0'),
        db.CheckConstraint('height >= 0'),
        db.CheckConstraint('width >= 0')
    )

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    path = db.Column(db.String(1024), nullable=False)
    frame = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)
    width = db.Column(db.Integer, nullable=False)
    data_id = db.Column(db.ForeignKey('engine_data.id', deferrable=True, initially='DEFERRED'), index=True)

    data = db.relationship('EngineDatum', primaryjoin='EngineImage.data_id == EngineDatum.id', backref='engine_images')



class EngineIssue(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_issue'
    __table_args__ = (
        db.CheckConstraint('frame >= 0'),
    )

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    frame = db.Column(db.Integer, nullable=False)
    position = db.Column(db.Text, nullable=False)
    created_date = db.Column(db.DateTime(True), nullable=False)
    updated_date = db.Column(db.DateTime(True))
    job_id = db.Column(db.ForeignKey('engine_job.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)
    owner_id = db.Column(db.ForeignKey('auth_user.id', deferrable=True, initially='DEFERRED'), index=True)
    assignee_id = db.Column(db.ForeignKey('auth_user.id', deferrable=True, initially='DEFERRED'), index=True)
    resolved = db.Column(db.Boolean, nullable=False)

    assignee = db.relationship('AuthUser', primaryjoin='EngineIssue.assignee_id == AuthUser.id', backref='authuser_engine_issues')
    job = db.relationship('EngineJob', primaryjoin='EngineIssue.job_id == EngineJob.id', backref='engine_issues')
    owner = db.relationship('AuthUser', primaryjoin='EngineIssue.owner_id == AuthUser.id', backref='authuser_engine_issues_0')



class EngineJob(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_job'

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    segment_id = db.Column(db.ForeignKey('engine_segment.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)
    assignee_id = db.Column(db.ForeignKey('auth_user.id', deferrable=True, initially='DEFERRED'), index=True)
    status = db.Column(db.String(32), nullable=False)
    stage = db.Column(db.String(32), nullable=False)
    state = db.Column(db.String(32), nullable=False)
    updated_date = db.Column(db.DateTime(True), nullable=False)

    assignee = db.relationship('AuthUser', primaryjoin='EngineJob.assignee_id == AuthUser.id', backref='engine_jobs')
    segment = db.relationship('EngineSegment', primaryjoin='EngineJob.segment_id == EngineSegment.id', backref='engine_jobs')



class EngineJobcommit(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_jobcommit'

    id = db.Column(db.BigInteger, primary_key=True, server_default=db.FetchedValue())
    timestamp = db.Column(db.DateTime(True), nullable=False)
    owner_id = db.Column(db.ForeignKey('auth_user.id', deferrable=True, initially='DEFERRED'), index=True)
    job_id = db.Column(db.ForeignKey('engine_job.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)
    data = db.Column(db.JSON, nullable=False)
    scope = db.Column(db.String(32), nullable=False)

    job = db.relationship('EngineJob', primaryjoin='EngineJobcommit.job_id == EngineJob.id', backref='engine_jobcommits')
    owner = db.relationship('AuthUser', primaryjoin='EngineJobcommit.owner_id == AuthUser.id', backref='engine_jobcommits')



class EngineLabel(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_label'
    __table_args__ = (
        db.UniqueConstraint('task_id', 'name', 'parent_id'),
    )

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    name = db.Column(db.String(64), nullable=False)
    task_id = db.Column(db.ForeignKey('engine_task.id', deferrable=True, initially='DEFERRED'), index=True)
    color = db.Column(db.String(8), nullable=False)
    project_id = db.Column(db.ForeignKey('engine_project.id', deferrable=True, initially='DEFERRED'), index=True)
    parent_id = db.Column(db.ForeignKey('engine_label.id', deferrable=True, initially='DEFERRED'), index=True)
    type = db.Column(db.String(32))

    parent = db.relationship('EngineLabel', remote_side=[id], primaryjoin='EngineLabel.parent_id == EngineLabel.id', backref='engine_labels')
    project = db.relationship('EngineProject', primaryjoin='EngineLabel.project_id == EngineProject.id', backref='engine_labels')
    task = db.relationship('EngineTask', primaryjoin='EngineLabel.task_id == EngineTask.id', backref='engine_labels')



class EngineLabeledimage(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_labeledimage'
    __table_args__ = (
        db.CheckConstraint('"group" >= 0'),
        db.CheckConstraint('frame >= 0')
    )

    id = db.Column(db.BigInteger, primary_key=True, server_default=db.FetchedValue())
    frame = db.Column(db.Integer, nullable=False)
    group = db.Column(db.Integer)
    job_id = db.Column(db.ForeignKey('engine_job.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)
    label_id = db.Column(db.ForeignKey('engine_label.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)
    source = db.Column(db.String(16))

    job = db.relationship('EngineJob', primaryjoin='EngineLabeledimage.job_id == EngineJob.id', backref='engine_labeledimages')
    label = db.relationship('EngineLabel', primaryjoin='EngineLabeledimage.label_id == EngineLabel.id', backref='engine_labeledimages')



class EngineLabeledimageattributeval(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_labeledimageattributeval'

    id = db.Column(db.BigInteger, primary_key=True, server_default=db.FetchedValue())
    value = db.Column(db.String(4096), nullable=False)
    spec_id = db.Column(db.ForeignKey('engine_attributespec.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)
    image_id = db.Column(db.ForeignKey('engine_labeledimage.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)

    image = db.relationship('EngineLabeledimage', primaryjoin='EngineLabeledimageattributeval.image_id == EngineLabeledimage.id', backref='engine_labeledimageattributevals')
    spec = db.relationship('EngineAttributespec', primaryjoin='EngineLabeledimageattributeval.spec_id == EngineAttributespec.id', backref='engine_labeledimageattributevals')



class EngineLabeledshape(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_labeledshape'
    __table_args__ = (
        db.CheckConstraint('"group" >= 0'),
        db.CheckConstraint('frame >= 0')
    )

    id = db.Column(db.BigInteger, primary_key=True, server_default=db.FetchedValue())
    frame = db.Column(db.Integer, nullable=False)
    group = db.Column(db.Integer)
    type = db.Column(db.String(16), nullable=False)
    occluded = db.Column(db.Boolean, nullable=False)
    z_order = db.Column(db.Integer, nullable=False)
    points = db.Column(db.Text, nullable=False)
    job_id = db.Column(db.ForeignKey('engine_job.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)
    label_id = db.Column(db.ForeignKey('engine_label.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)
    source = db.Column(db.String(16))
    rotation = db.Column(db.Double(53), nullable=False)
    parent_id = db.Column(db.ForeignKey('engine_labeledshape.id', deferrable=True, initially='DEFERRED'), index=True)
    outside = db.Column(db.Boolean, nullable=False)

    job = db.relationship('EngineJob', primaryjoin='EngineLabeledshape.job_id == EngineJob.id', backref='engine_labeledshapes')
    label = db.relationship('EngineLabel', primaryjoin='EngineLabeledshape.label_id == EngineLabel.id', backref='engine_labeledshapes')
    parent = db.relationship('EngineLabeledshape', remote_side=[id], primaryjoin='EngineLabeledshape.parent_id == EngineLabeledshape.id', backref='engine_labeledshapes')



class EngineLabeledshapeattributeval(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_labeledshapeattributeval'

    id = db.Column(db.BigInteger, primary_key=True, server_default=db.FetchedValue())
    value = db.Column(db.String(4096), nullable=False)
    spec_id = db.Column(db.ForeignKey('engine_attributespec.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)
    shape_id = db.Column(db.ForeignKey('engine_labeledshape.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)

    shape = db.relationship('EngineLabeledshape', primaryjoin='EngineLabeledshapeattributeval.shape_id == EngineLabeledshape.id', backref='engine_labeledshapeattributevals')
    spec = db.relationship('EngineAttributespec', primaryjoin='EngineLabeledshapeattributeval.spec_id == EngineAttributespec.id', backref='engine_labeledshapeattributevals')



class EngineLabeledtrack(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_labeledtrack'
    __table_args__ = (
        db.CheckConstraint('"group" >= 0'),
        db.CheckConstraint('frame >= 0')
    )

    id = db.Column(db.BigInteger, primary_key=True, server_default=db.FetchedValue())
    frame = db.Column(db.Integer, nullable=False)
    group = db.Column(db.Integer)
    job_id = db.Column(db.ForeignKey('engine_job.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)
    label_id = db.Column(db.ForeignKey('engine_label.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)
    source = db.Column(db.String(16))
    parent_id = db.Column(db.ForeignKey('engine_labeledtrack.id', deferrable=True, initially='DEFERRED'), index=True)

    job = db.relationship('EngineJob', primaryjoin='EngineLabeledtrack.job_id == EngineJob.id', backref='engine_labeledtracks')
    label = db.relationship('EngineLabel', primaryjoin='EngineLabeledtrack.label_id == EngineLabel.id', backref='engine_labeledtracks')
    parent = db.relationship('EngineLabeledtrack', remote_side=[id], primaryjoin='EngineLabeledtrack.parent_id == EngineLabeledtrack.id', backref='engine_labeledtracks')



class EngineLabeledtrackattributeval(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_labeledtrackattributeval'

    id = db.Column(db.BigInteger, primary_key=True, server_default=db.FetchedValue())
    value = db.Column(db.String(4096), nullable=False)
    spec_id = db.Column(db.ForeignKey('engine_attributespec.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)
    track_id = db.Column(db.ForeignKey('engine_labeledtrack.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)

    spec = db.relationship('EngineAttributespec', primaryjoin='EngineLabeledtrackattributeval.spec_id == EngineAttributespec.id', backref='engine_labeledtrackattributevals')
    track = db.relationship('EngineLabeledtrack', primaryjoin='EngineLabeledtrackattributeval.track_id == EngineLabeledtrack.id', backref='engine_labeledtrackattributevals')



class EngineManifest(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_manifest'

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    filename = db.Column(db.String(1024), nullable=False)
    cloud_storage_id = db.Column(db.ForeignKey('engine_cloudstorage.id', deferrable=True, initially='DEFERRED'), index=True)

    cloud_storage = db.relationship('EngineCloudstorage', primaryjoin='EngineManifest.cloud_storage_id == EngineCloudstorage.id', backref='engine_manifests')



class EngineProfile(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_profile'

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    rating = db.Column(db.Double(53), nullable=False)
    user_id = db.Column(db.ForeignKey('auth_user.id', deferrable=True, initially='DEFERRED'), nullable=False, unique=True)

    user = db.relationship('AuthUser', uselist=False, primaryjoin='EngineProfile.user_id == AuthUser.id', backref='engine_profiles')



class EngineProject(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_project'

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    name = db.Column(db.String(256), nullable=False)
    bug_tracker = db.Column(db.String(2000), nullable=False)
    created_date = db.Column(db.DateTime(True), nullable=False)
    updated_date = db.Column(db.DateTime(True), nullable=False)
    status = db.Column(db.String(32), nullable=False)
    assignee_id = db.Column(db.ForeignKey('auth_user.id', deferrable=True, initially='DEFERRED'), index=True)
    owner_id = db.Column(db.ForeignKey('auth_user.id', deferrable=True, initially='DEFERRED'), index=True)
    organization_id = db.Column(db.ForeignKey('organizations_organization.id', deferrable=True, initially='DEFERRED'), index=True)
    source_storage_id = db.Column(db.ForeignKey('engine_storage.id', deferrable=True, initially='DEFERRED'), index=True)
    target_storage_id = db.Column(db.ForeignKey('engine_storage.id', deferrable=True, initially='DEFERRED'), index=True)

    assignee = db.relationship('AuthUser', primaryjoin='EngineProject.assignee_id == AuthUser.id', backref='authuser_engine_projects')
    organization = db.relationship('OrganizationsOrganization', primaryjoin='EngineProject.organization_id == OrganizationsOrganization.id', backref='engine_projects')
    owner = db.relationship('AuthUser', primaryjoin='EngineProject.owner_id == AuthUser.id', backref='authuser_engine_projects_0')
    source_storage = db.relationship('EngineStorage', primaryjoin='EngineProject.source_storage_id == EngineStorage.id', backref='enginestorage_engine_projects')
    target_storage = db.relationship('EngineStorage', primaryjoin='EngineProject.target_storage_id == EngineStorage.id', backref='enginestorage_engine_projects_0')



class EngineRelatedfile(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_relatedfile'
    __table_args__ = (
        db.UniqueConstraint('data_id', 'path'),
    )

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    path = db.Column(db.String(1024), nullable=False)
    data_id = db.Column(db.ForeignKey('engine_data.id', deferrable=True, initially='DEFERRED'), index=True)
    primary_image_id = db.Column(db.ForeignKey('engine_image.id', deferrable=True, initially='DEFERRED'), index=True)

    data = db.relationship('EngineDatum', primaryjoin='EngineRelatedfile.data_id == EngineDatum.id', backref='engine_relatedfiles')
    primary_image = db.relationship('EngineImage', primaryjoin='EngineRelatedfile.primary_image_id == EngineImage.id', backref='engine_relatedfiles')



class EngineRemotefile(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_remotefile'

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    file = db.Column(db.String(1024), nullable=False)
    data_id = db.Column(db.ForeignKey('engine_data.id', deferrable=True, initially='DEFERRED'), index=True)

    data = db.relationship('EngineDatum', primaryjoin='EngineRemotefile.data_id == EngineDatum.id', backref='engine_remotefiles')



class EngineSegment(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_segment'

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    start_frame = db.Column(db.Integer, nullable=False)
    stop_frame = db.Column(db.Integer, nullable=False)
    task_id = db.Column(db.ForeignKey('engine_task.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)

    task = db.relationship('EngineTask', primaryjoin='EngineSegment.task_id == EngineTask.id', backref='engine_segments')



class EngineServerfile(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_serverfile'

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    file = db.Column(db.String(1024), nullable=False)
    data_id = db.Column(db.ForeignKey('engine_data.id', deferrable=True, initially='DEFERRED'), index=True)

    data = db.relationship('EngineDatum', primaryjoin='EngineServerfile.data_id == EngineDatum.id', backref='engine_serverfiles')



class EngineSkeleton(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_skeleton'

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    svg = db.Column(db.Text)
    root_id = db.Column(db.ForeignKey('engine_label.id', deferrable=True, initially='DEFERRED'), nullable=False, unique=True)

    root = db.relationship('EngineLabel', uselist=False, primaryjoin='EngineSkeleton.root_id == EngineLabel.id', backref='engine_skeletons')



class EngineStorage(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_storage'

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    location = db.Column(db.String(16), nullable=False)
    cloud_storage_id = db.Column(db.Integer)



class EngineTask(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_task'
    __table_args__ = (
        db.CheckConstraint('overlap >= 0'),
        db.CheckConstraint('segment_size >= 0')
    )

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    name = db.Column(db.String(256), nullable=False)
    mode = db.Column(db.String(32), nullable=False)
    created_date = db.Column(db.DateTime(True), nullable=False)
    updated_date = db.Column(db.DateTime(True), nullable=False)
    status = db.Column(db.String(32), nullable=False)
    bug_tracker = db.Column(db.String(2000), nullable=False)
    owner_id = db.Column(db.ForeignKey('auth_user.id', deferrable=True, initially='DEFERRED'), index=True)
    overlap = db.Column(db.Integer)
    assignee_id = db.Column(db.ForeignKey('auth_user.id', deferrable=True, initially='DEFERRED'), index=True)
    segment_size = db.Column(db.Integer, nullable=False)
    project_id = db.Column(db.ForeignKey('engine_project.id', deferrable=True, initially='DEFERRED'), index=True)
    data_id = db.Column(db.ForeignKey('engine_data.id', deferrable=True, initially='DEFERRED'), index=True)
    dimension = db.Column(db.String(2), nullable=False)
    subset = db.Column(db.String(64), nullable=False)
    organization_id = db.Column(db.ForeignKey('organizations_organization.id', deferrable=True, initially='DEFERRED'), index=True)
    source_storage_id = db.Column(db.ForeignKey('engine_storage.id', deferrable=True, initially='DEFERRED'), index=True)
    target_storage_id = db.Column(db.ForeignKey('engine_storage.id', deferrable=True, initially='DEFERRED'), index=True)

    assignee = db.relationship('AuthUser', primaryjoin='EngineTask.assignee_id == AuthUser.id', backref='authuser_engine_tasks')
    data = db.relationship('EngineDatum', primaryjoin='EngineTask.data_id == EngineDatum.id', backref='engine_tasks')
    organization = db.relationship('OrganizationsOrganization', primaryjoin='EngineTask.organization_id == OrganizationsOrganization.id', backref='engine_tasks')
    owner = db.relationship('AuthUser', primaryjoin='EngineTask.owner_id == AuthUser.id', backref='authuser_engine_tasks_0')
    project = db.relationship('EngineProject', primaryjoin='EngineTask.project_id == EngineProject.id', backref='engine_tasks')
    source_storage = db.relationship('EngineStorage', primaryjoin='EngineTask.source_storage_id == EngineStorage.id', backref='enginestorage_engine_tasks')
    target_storage = db.relationship('EngineStorage', primaryjoin='EngineTask.target_storage_id == EngineStorage.id', backref='enginestorage_engine_tasks_0')


class DatasetRepoGitdatum(EngineTask):
    __tablename__ = 'dataset_repo_gitdata'

    task_id = db.Column(db.ForeignKey('engine_task.id', deferrable=True, initially='DEFERRED'), primary_key=True)
    url = db.Column(db.String(2000), nullable=False)
    path = db.Column(db.String(256), nullable=False)
    sync_date = db.Column(db.DateTime(True), nullable=False)
    status = db.Column(db.String(20), nullable=False)
    lfs = db.Column(db.Boolean, nullable=False)
    format = db.Column(db.String(256), nullable=False)



class EngineTrackedshape(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_trackedshape'
    __table_args__ = (
        db.CheckConstraint('frame >= 0'),
    )

    type = db.Column(db.String(16), nullable=False)
    occluded = db.Column(db.Boolean, nullable=False)
    z_order = db.Column(db.Integer, nullable=False)
    points = db.Column(db.Text, nullable=False)
    id = db.Column(db.BigInteger, primary_key=True, server_default=db.FetchedValue())
    frame = db.Column(db.Integer, nullable=False)
    outside = db.Column(db.Boolean, nullable=False)
    track_id = db.Column(db.ForeignKey('engine_labeledtrack.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)
    rotation = db.Column(db.Double(53), nullable=False)

    track = db.relationship('EngineLabeledtrack', primaryjoin='EngineTrackedshape.track_id == EngineLabeledtrack.id', backref='engine_trackedshapes')



class EngineTrackedshapeattributeval(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_trackedshapeattributeval'

    id = db.Column(db.BigInteger, primary_key=True, server_default=db.FetchedValue())
    value = db.Column(db.String(4096), nullable=False)
    shape_id = db.Column(db.ForeignKey('engine_trackedshape.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)
    spec_id = db.Column(db.ForeignKey('engine_attributespec.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)

    shape = db.relationship('EngineTrackedshape', primaryjoin='EngineTrackedshapeattributeval.shape_id == EngineTrackedshape.id', backref='engine_trackedshapeattributevals')
    spec = db.relationship('EngineAttributespec', primaryjoin='EngineTrackedshapeattributeval.spec_id == EngineAttributespec.id', backref='engine_trackedshapeattributevals')



class EngineVideo(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'engine_video'
    __table_args__ = (
        db.CheckConstraint('height >= 0'),
        db.CheckConstraint('width >= 0')
    )

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    path = db.Column(db.String(1024), nullable=False)
    height = db.Column(db.Integer, nullable=False)
    width = db.Column(db.Integer, nullable=False)
    data_id = db.Column(db.ForeignKey('engine_data.id', deferrable=True, initially='DEFERRED'), unique=True)

    data = db.relationship('EngineDatum', uselist=False, primaryjoin='EngineVideo.data_id == EngineDatum.id', backref='engine_videos')



class OrganizationsInvitation(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'organizations_invitation'

    key = db.Column(db.String(64), primary_key=True, index=True)
    created_date = db.Column(db.DateTime(True), nullable=False)
    membership_id = db.Column(db.ForeignKey('organizations_membership.id', deferrable=True, initially='DEFERRED'), nullable=False, unique=True)
    owner_id = db.Column(db.ForeignKey('auth_user.id', deferrable=True, initially='DEFERRED'), index=True)

    membership = db.relationship('OrganizationsMembership', uselist=False, primaryjoin='OrganizationsInvitation.membership_id == OrganizationsMembership.id', backref='organizations_invitations')
    owner = db.relationship('AuthUser', primaryjoin='OrganizationsInvitation.owner_id == AuthUser.id', backref='organizations_invitations')



class OrganizationsMembership(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'organizations_membership'
    __table_args__ = (
        db.UniqueConstraint('user_id', 'organization_id'),
    )

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    is_active = db.Column(db.Boolean, nullable=False)
    joined_date = db.Column(db.DateTime(True))
    role = db.Column(db.String(16), nullable=False)
    organization_id = db.Column(db.ForeignKey('organizations_organization.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)
    user_id = db.Column(db.ForeignKey('auth_user.id', deferrable=True, initially='DEFERRED'), index=True)

    organization = db.relationship('OrganizationsOrganization', primaryjoin='OrganizationsMembership.organization_id == OrganizationsOrganization.id', backref='organizations_memberships')
    user = db.relationship('AuthUser', primaryjoin='OrganizationsMembership.user_id == AuthUser.id', backref='organizations_memberships')



class OrganizationsOrganization(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'organizations_organization'

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    slug = db.Column(db.String(16), nullable=False, unique=True)
    name = db.Column(db.String(64), nullable=False)
    description = db.Column(db.Text, nullable=False)
    created_date = db.Column(db.DateTime(True), nullable=False)
    updated_date = db.Column(db.DateTime(True), nullable=False)
    contact = db.Column(db.JSON, nullable=False)
    owner_id = db.Column(db.ForeignKey('auth_user.id', deferrable=True, initially='DEFERRED'), index=True)

    owner = db.relationship('AuthUser', primaryjoin='OrganizationsOrganization.owner_id == AuthUser.id', backref='organizations_organizations')



class SocialaccountSocialaccount(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'socialaccount_socialaccount'
    __table_args__ = (
        db.UniqueConstraint('provider', 'uid'),
    )

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    provider = db.Column(db.String(30), nullable=False)
    uid = db.Column(db.String(191), nullable=False)
    last_login = db.Column(db.DateTime(True), nullable=False)
    date_joined = db.Column(db.DateTime(True), nullable=False)
    extra_data = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.ForeignKey('auth_user.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)

    user = db.relationship('AuthUser', primaryjoin='SocialaccountSocialaccount.user_id == AuthUser.id', backref='socialaccount_socialaccounts')



class SocialaccountSocialapp(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'socialaccount_socialapp'

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    provider = db.Column(db.String(30), nullable=False)
    name = db.Column(db.String(40), nullable=False)
    client_id = db.Column(db.String(191), nullable=False)
    secret = db.Column(db.String(191), nullable=False)
    key = db.Column(db.String(191), nullable=False)



class SocialaccountSocialappSite(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'socialaccount_socialapp_sites'
    __table_args__ = (
        db.UniqueConstraint('socialapp_id', 'site_id'),
    )

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    socialapp_id = db.Column(db.ForeignKey('socialaccount_socialapp.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)
    site_id = db.Column(db.ForeignKey('django_site.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)

    site = db.relationship('DjangoSite', primaryjoin='SocialaccountSocialappSite.site_id == DjangoSite.id', backref='socialaccount_socialapp_sites')
    socialapp = db.relationship('SocialaccountSocialapp', primaryjoin='SocialaccountSocialappSite.socialapp_id == SocialaccountSocialapp.id', backref='socialaccount_socialapp_sites')



class SocialaccountSocialtoken(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'socialaccount_socialtoken'
    __table_args__ = (
        db.UniqueConstraint('app_id', 'account_id'),
    )

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    token = db.Column(db.Text, nullable=False)
    token_secret = db.Column(db.Text, nullable=False)
    expires_at = db.Column(db.DateTime(True))
    account_id = db.Column(db.ForeignKey('socialaccount_socialaccount.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)
    app_id = db.Column(db.ForeignKey('socialaccount_socialapp.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)

    account = db.relationship('SocialaccountSocialaccount', primaryjoin='SocialaccountSocialtoken.account_id == SocialaccountSocialaccount.id', backref='socialaccount_socialtokens')
    app = db.relationship('SocialaccountSocialapp', primaryjoin='SocialaccountSocialtoken.app_id == SocialaccountSocialapp.id', backref='socialaccount_socialtokens')



class WebhooksWebhook(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'webhooks_webhook'
    __table_args__ = (
        db.CheckConstraint("project_id IS NOT NULL AND type::text = 'project'::text OR organization_id IS NOT NULL AND project_id IS NULL AND type::text = 'organization'::text"),
    )

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    target_url = db.Column(db.String(200), nullable=False)
    description = db.Column(db.String(128), nullable=False)
    events = db.Column(db.String(4096), nullable=False)
    type = db.Column(db.String(16), nullable=False)
    content_type = db.Column(db.String(64), nullable=False)
    secret = db.Column(db.String(64), nullable=False)
    is_active = db.Column(db.Boolean, nullable=False)
    enable_ssl = db.Column(db.Boolean, nullable=False)
    created_date = db.Column(db.DateTime(True), nullable=False)
    updated_date = db.Column(db.DateTime(True), nullable=False)
    organization_id = db.Column(db.ForeignKey('organizations_organization.id', deferrable=True, initially='DEFERRED'), index=True)
    owner_id = db.Column(db.ForeignKey('auth_user.id', deferrable=True, initially='DEFERRED'), index=True)
    project_id = db.Column(db.ForeignKey('engine_project.id', deferrable=True, initially='DEFERRED'), index=True)

    organization = db.relationship('OrganizationsOrganization', primaryjoin='WebhooksWebhook.organization_id == OrganizationsOrganization.id', backref='webhooks_webhooks')
    owner = db.relationship('AuthUser', primaryjoin='WebhooksWebhook.owner_id == AuthUser.id', backref='webhooks_webhooks')
    project = db.relationship('EngineProject', primaryjoin='WebhooksWebhook.project_id == EngineProject.id', backref='webhooks_webhooks')



class WebhooksWebhookdelivery(db.Model):
    __bind_key__ = 'cvat'
    __tablename__ = 'webhooks_webhookdelivery'
    __table_args__ = (
        db.CheckConstraint('status_code >= 0'),
    )

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    event = db.Column(db.String(64), nullable=False)
    status_code = db.Column(db.Integer)
    redelivery = db.Column(db.Boolean, nullable=False)
    created_date = db.Column(db.DateTime(True), nullable=False)
    updated_date = db.Column(db.DateTime(True), nullable=False)
    changed_fields = db.Column(db.String(4096), nullable=False)
    request = db.Column(db.JSON, nullable=False)
    response = db.Column(db.JSON, nullable=False)
    webhook_id = db.Column(db.ForeignKey('webhooks_webhook.id', deferrable=True, initially='DEFERRED'), nullable=False, index=True)

    webhook = db.relationship('WebhooksWebhook', primaryjoin='WebhooksWebhookdelivery.webhook_id == WebhooksWebhook.id', backref='webhooks_webhookdeliveries')
