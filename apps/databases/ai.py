from apps.databases import db



class AiModelInfo(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'ai_model_info'
    __table_args__ = {'schema': 'amr'}

    model_id = db.Column(db.Integer, primary_key=True, nullable=False, server_default=db.FetchedValue())
    model_type = db.Column(db.String(20))
    model_name = db.Column(db.Text)
    model_path = db.Column(db.Text)
    model_version = db.Column(db.Text)
    comp_category = db.Column(db.Text)
    defect_category = db.Column(db.Text)
    verified_status = db.Column(db.String(20), nullable=False, server_default=db.FetchedValue(), info='模型驗證')
    create_time = db.Column(db.DateTime(True), primary_key=True, nullable=False, server_default=db.FetchedValue())
    finetune_id = db.Column(db.Integer, nullable=False)
    finetune_type = db.Column(db.String, nullable=False)
    training_verified_status = db.Column(db.Text, info='訓練驗證驗證')



class AiModelPerf(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'ai_model_perf'
    __table_args__ = {'schema': 'amr'}

    model_id = db.Column(db.Integer, primary_key=True)
    metrics_result = db.Column(db.Text, nullable=False)
    false_negative_imgs = db.Column(db.Text, nullable=False)
    false_positive_imgs = db.Column(db.Text, nullable=False)
    insert_time = db.Column(db.DateTime(True), nullable=False, server_default=db.FetchedValue())



class AiServerInfo(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'ai_server_info'
    __table_args__ = {'schema': 'amr'}

    server_id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    server_name = db.Column(db.String(30))
    server_type = db.Column(db.String(30))
    server_url = db.Column(db.Text)



class AmrModelSlot(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'amr_model_slot'
    __table_args__ = {'schema': 'amr'}

    slot_id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    comp_category = db.Column(db.String, nullable=False)
    defect_category = db.Column(db.String, nullable=False)
    model_func = db.Column(db.String, nullable=False)



class AmrPositionDeploy(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'amr_position_deploy'
    __table_args__ = {'schema': 'amr'}

    position_id = db.Column(db.ForeignKey('amr.amr_position_info.position_id'), primary_key=True, nullable=False)
    slot_id = db.Column(db.ForeignKey('amr.amr_model_slot.slot_id'), primary_key=True, nullable=False)
    model_id = db.Column(db.Integer, primary_key=True, nullable=False)
    create_time = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())

    position = db.relationship('AmrPositionInfo', primaryjoin='AmrPositionDeploy.position_id == AmrPositionInfo.position_id', backref='amr_position_deploys')
    slot = db.relationship('AmrModelSlot', primaryjoin='AmrPositionDeploy.slot_id == AmrModelSlot.slot_id', backref='amr_position_deploys')



class AmrPositionInfo(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'amr_position_info'
    __table_args__ = {'schema': 'amr'}

    position_id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    site = db.Column(db.String)
    line = db.Column(db.String)



class AugImg(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'aug_img'
    __table_args__ = {'schema': 'amr'}

    aug_img_id = db.Column(db.BigInteger, primary_key=True, nullable=False, server_default=db.FetchedValue())
    img_category_id = db.Column(db.Integer, primary_key=True, nullable=False)
    src_img_id = db.Column(db.BigInteger, primary_key=True, nullable=False)
    aug_method = db.Column(db.String(15))
    img_create_time = db.Column(db.DateTime(True), primary_key=True, nullable=False, server_default=db.FetchedValue())
    insert_time = db.Column(db.DateTime(True), primary_key=True, nullable=False, server_default=db.FetchedValue())



class CategoryMapping(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'category_mapping'
    __table_args__ = {'schema': 'amr'}

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    site = db.Column(db.String, nullable=False)
    factory = db.Column(db.String, nullable=False)
    line = db.Column(db.String, nullable=False)
    group_type = db.Column(db.String, nullable=False)
    ng_category = db.Column(db.String, nullable=False)
    ok_category = db.Column(db.String, nullable=False)
    project = db.Column(db.String, nullable=False)
    labels = db.Column(db.String, nullable=False)
    mode = db.Column(db.Integer, nullable=False)
    od_model = db.Column(db.String)
    od_mapping_category = db.Column(db.String)
    cls_model = db.Column(db.String, nullable=False)
    cls_mapping_category = db.Column(db.String, nullable=False)



class ClsTrainingInfo(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'cls_training_info'
    __table_args__ = {'schema': 'amr'}

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    datetime = db.Column(db.DateTime(True), nullable=False, server_default=db.FetchedValue())
    task_id = db.Column(db.Integer, nullable=False)
    comp_type = db.Column(db.String(32), nullable=False)
    val_status = db.Column(db.String(32), nullable=False)
    model_version = db.Column(db.String(20))
    category_mapping = db.Column(db.Text)



class CriticalNg(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'critical_ng'
    __table_args__ = {'schema': 'amr'}

    img_id = db.Column(db.Text, primary_key=True)
    line_id = db.Column(db.Text, nullable=False, server_default=db.FetchedValue())
    image_path = db.Column(db.Text, nullable=False)
    group_type = db.Column(db.Text, nullable=False)
    cls_model = db.Column(db.Text, nullable=False, server_default=db.FetchedValue())



class CropCategorizingRecord(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'crop_categorizing_record'
    __table_args__ = {'schema': 'amr'}

    finetune_id = db.Column(db.Integer, primary_key=True, nullable=False)
    img_id = db.Column(db.Text, primary_key=True, nullable=False, info='原圖 ID')
    crop_name = db.Column(db.Text, primary_key=True, nullable=False)
    x_min = db.Column(db.Integer, nullable=False)
    y_min = db.Column(db.Integer, nullable=False)
    x_max = db.Column(db.Integer, nullable=False)
    y_max = db.Column(db.Integer, nullable=False)
    categorizing_code = db.Column(db.String(10), nullable=False, server_default=db.FetchedValue(), info='分類結果')
    update_time = db.Column(db.DateTime(True), nullable=False, server_default=db.FetchedValue(), info='更新時間')
    create_time = db.Column(db.DateTime(True), nullable=False, server_default=db.FetchedValue(), info='建立時間')
    finetune_type = db.Column(db.String, primary_key=True, nullable=False, info='1:IRI 2:URD')
    critical_ng = db.Column(db.Boolean, nullable=False, server_default=db.FetchedValue())



class CropImg(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'crop_img'
    __table_args__ = {'schema': 'amr'}

    crop_img_id = db.Column(db.BigInteger, primary_key=True, nullable=False, server_default=db.FetchedValue())
    src_img_id = db.Column(db.BigInteger, primary_key=True, nullable=False)
    img_length = db.Column(db.Integer)
    img_width = db.Column(db.Integer)
    img_create_time = db.Column(db.DateTime(True), primary_key=True, nullable=False, server_default=db.FetchedValue())
    insert_time = db.Column(db.DateTime(True), primary_key=True, nullable=False, server_default=db.FetchedValue())



class DefectCategoryInfo(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'defect_category_info'
    __table_args__ = {'schema': 'amr'}

    defect_id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    detection_code = db.Column(db.String(10))
    defect_category = db.Column(db.String(20))
    defect_type = db.Column(db.Text)



class DetectionRecord(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'detection_record'
    __table_args__ = {'schema': 'amr'}

    img_category_id = db.Column(db.Integer, primary_key=True, nullable=False, info='圖片類型代碼2 =CROP')
    src_img_id = db.Column(db.BigInteger, primary_key=True, nullable=False, info='原圖ID')
    inspector_id = db.Column(db.Integer, primary_key=True, nullable=False, info='代碼 1 = ai 2= human')
    defect_category = db.Column(db.String(30), primary_key=True, nullable=False, info='新架構後沒用到')
    detection_code = db.Column(db.String(10), nullable=False, server_default=db.FetchedValue(), info='分類結果')
    inspect_time = db.Column(db.DateTime(True), primary_key=True, nullable=False, server_default=db.FetchedValue())
    insert_time = db.Column(db.DateTime(True), primary_key=True, nullable=False, server_default=db.FetchedValue())



class EnvironmentVariable(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'environment_variable'
    __table_args__ = {'schema': 'amr'}

    id = db.Column(db.Text, primary_key=True)
    value = db.Column(db.Text)



class ImagePool(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'image_pool'
    __table_args__ = {'schema': 'amr'}

    line = db.Column(db.Text, nullable=False)
    ip = db.Column(db.Text, nullable=False)
    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    prefix = db.Column(db.Text)



class ImgCategoryInfo(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'img_category_info'
    __table_args__ = {'schema': 'amr'}

    img_type_id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    img_category = db.Column(db.String(15))



class ImgPath(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'img_path'
    __table_args__ = {'schema': 'amr'}

    img_category_id = db.Column(db.Integer, primary_key=True, nullable=False)
    src_img_id = db.Column(db.BigInteger, primary_key=True, nullable=False)
    server_id = db.Column(db.Integer, primary_key=True, nullable=False)
    img_file_path = db.Column(db.Text)
    create_time = db.Column(db.DateTime(True), primary_key=True, nullable=False, server_default=db.FetchedValue())



class InspectorInfo(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'inspector_info'
    __table_args__ = {'schema': 'amr'}

    inspector_id = db.Column(db.Integer, primary_key=True, nullable=False, server_default=db.FetchedValue())
    inspector_type = db.Column(db.String(10))
    inspector_name = db.Column(db.String(30))
    create_time = db.Column(db.DateTime(True), primary_key=True, nullable=False, server_default=db.FetchedValue())



class IriRecord(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'iri_record'
    __table_args__ = {'schema': 'amr'}

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue(), info='No.')
    project = db.Column(db.Text, info='CVAT project name')
    task = db.Column(db.Text, info='CVAT task name')
    status = db.Column(db.Text)
    site = db.Column(db.String)
    line = db.Column(db.String)
    group_type = db.Column(db.String)
    start_date = db.Column(db.DateTime(True), info='Time From')
    end_date = db.Column(db.DateTime(True), info='Time To')
    labeling = db.Column(db.Boolean)
    od_training = db.Column(db.Text)
    cls_training = db.Column(db.Text)
    update_time = db.Column(db.DateTime(True))
    create_time = db.Column(db.DateTime(True), nullable=False, server_default=db.FetchedValue())
    task_id = db.Column(db.Integer, info='CVAT task_id')
    project_id = db.Column(db.Integer, info='CVAT project_id')
    smart_filter = db.Column(db.Boolean, server_default=db.FetchedValue())
    images = db.Column(db.Text)
    image_mode = db.Column(db.Text)
    creator = db.Column(db.Text, info='建立者')
    smart_filter_value = db.Column(db.Integer)



class ModelManagement(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'model_management'
    __table_args__ = {'schema': 'amr'}

    training_id = db.Column(db.Text, primary_key=True)
    model_type = db.Column(db.Text, nullable=False)
    task = db.Column(db.Text, nullable=False)
    fantout_time = db.Column(db.DateTime(True))
    create_user = db.Column(db.Text, nullable=False)
    approved_user = db.Column(db.Text, nullable=False)
    fanout_user = db.Column(db.Text, nullable=False)
    result_states = db.Column(db.Text, nullable=False)
    fanout_line = db.Column(db.Text)
    roll_back_line = db.Column(db.Text)
    create_time = db.Column(db.DateTime(True), nullable=False)



class OdTrainingInfo(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'od_training_info'
    __table_args__ = {'schema': 'amr'}

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    datetime = db.Column(db.DateTime(True), nullable=False, server_default=db.FetchedValue())
    task_id = db.Column(db.Integer, nullable=False)
    comp_type = db.Column(db.String(32), nullable=False)
    val_status = db.Column(db.String(32), nullable=False)
    model_version = db.Column(db.String(20))
    category_mapping = db.Column(db.Text)



class OriginCategorizingRecord(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'origin_categorizing_record'
    __table_args__ = {'schema': 'amr'}

    finetune_id = db.Column(db.Integer, primary_key=True, nullable=False)
    img_id = db.Column(db.Text, primary_key=True, nullable=False)
    categorizing_code = db.Column(db.String(10), nullable=False, server_default=db.FetchedValue())
    update_time = db.Column(db.DateTime(True), nullable=False, server_default=db.FetchedValue())
    create_time = db.Column(db.DateTime(True), nullable=False, server_default=db.FetchedValue())



class OriginImg(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'origin_img'
    __table_args__ = {'schema': 'amr'}

    img_id = db.Column(db.BigInteger, primary_key=True, nullable=False, server_default=db.FetchedValue())
    site = db.Column(db.String(5), primary_key=True, nullable=False)
    line = db.Column(db.String(10), primary_key=True, nullable=False)
    product_name = db.Column(db.String(20))
    carrier_code = db.Column(db.String)
    comp_category = db.Column(db.String)
    comp_type = db.Column(db.String)
    comp_name = db.Column(db.String)
    light_type = db.Column(db.String)
    img_create_time = db.Column(db.DateTime(True), primary_key=True, nullable=False, server_default=db.FetchedValue())
    insert_time = db.Column(db.DateTime(True), primary_key=True, nullable=False, server_default=db.FetchedValue())



class Polarity(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'polarity'
    __table_args__ = {'schema': 'amr'}

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    product = db.Column(db.String(64), nullable=False)
    category = db.Column(db.Text)
    component_name = db.Column(db.Text)
    model_type = db.Column(db.String(16))
    model_name = db.Column(db.String(64), nullable=False)
    model_path = db.Column(db.String(526))
    active = db.Column(db.Boolean, nullable=False)
    version = db.Column(db.String(16), nullable=False)
    site = db.Column(db.String(16), nullable=False)
    line = db.Column(db.String(16), nullable=False)
    create_time = db.Column(db.DateTime(True), nullable=False, server_default=db.FetchedValue())
    update_time = db.Column(db.DateTime(True))



class TrainingSchedule(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'training_schedule'
    __table_args__ = {'schema': 'amr'}

    sched_id = db.Column(db.Integer, primary_key=True, nullable=False, server_default=db.FetchedValue())
    training_settings = db.Column(db.Text, nullable=False)
    dataset_request = db.Column(db.Text, nullable=False)
    sched_status = db.Column(db.String(20), nullable=False, server_default=db.FetchedValue())
    sched_insert_time = db.Column(db.DateTime(True), primary_key=True, nullable=False, server_default=db.FetchedValue())
    training_start = db.Column(db.DateTime(True))
    training_end = db.Column(db.DateTime(True))



class UploadData(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'upload_data'
    __table_args__ = {'schema': 'amr'}

    uuid = db.Column(db.Text, primary_key=True)
    site = db.Column(db.Text, nullable=False)
    factory = db.Column(db.Text, nullable=False)
    line_id = db.Column(db.Text)
    part_number = db.Column(db.Text)
    group_type = db.Column(db.Text)
    create_time = db.Column(db.DateTime, nullable=False)
    image_path = db.Column(db.Text)



class UrdRecord(db.Model):
    __bind_key__ = 'ai'
    __tablename__ = 'urd_record'
    __table_args__ = {'schema': 'amr'}

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    project = db.Column(db.Text)
    task = db.Column(db.Text)
    status = db.Column(db.Text)
    site = db.Column(db.Text)
    line = db.Column(db.Text)
    group_type = db.Column(db.Text)
    start_date = db.Column(db.DateTime(True))
    end_date = db.Column(db.DateTime(True))
    labeling = db.Column(db.Boolean)
    od_training = db.Column(db.Text)
    cls_training = db.Column(db.Text)
    update_time = db.Column(db.DateTime(True))
    create_time = db.Column(db.DateTime(True), nullable=False, server_default=db.FetchedValue())
    task_id = db.Column(db.Integer)
    project_id = db.Column(db.Integer)
    categorizing = db.Column(db.Boolean)
    category_ready = db.Column(db.Boolean)
    image_mode = db.Column(db.Text, info='query or upload')
    images = db.Column(db.Text)
    smart_filter = db.Column(db.Boolean, server_default=db.FetchedValue())
    creator = db.Column(db.Text, info='建立者')
    smart_filter_value = db.Column(db.Integer)
