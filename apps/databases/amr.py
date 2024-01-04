from apps.databases import db



class AmrFailLog(db.Model):
    __bind_key__ = 'amr'
    __tablename__ = 'amr_fail_log'
    __table_args__ = {'schema': 'public'}

    uuid = db.Column(db.Text, primary_key=True, nullable=False)
    datetime = db.Column(db.DateTime, primary_key=True, nullable=False)
    site = db.Column(db.String(4), primary_key=True, nullable=False)
    factory = db.Column(db.String(4), primary_key=True, nullable=False)
    line = db.Column(db.String(8), nullable=False)
    aoi = db.Column(db.Text, nullable=False)
    product = db.Column(db.Text, nullable=False)
    aoi_alarm = db.Column(db.Integer)
    amr_alarm = db.Column(db.Integer)
    rs_ng = db.Column(db.Integer)
    defect_type = db.Column(db.Text)
    component_type = db.Column(db.Text)
    covered_quantity = db.Column(db.Integer)
    uncovered_quantity = db.Column(db.Integer)
    amr_ok = db.Column(db.Integer)
    covered_or_not = db.Column(db.Boolean)
    component_group = db.Column(db.Text)
    carrier_count = db.Column(db.Integer)
    board_count = db.Column(db.Integer)
    station = db.Column(db.Text)



class AmrPassLog(db.Model):
    __bind_key__ = 'amr'
    __tablename__ = 'amr_pass_log'
    __table_args__ = {'schema': 'public'}

    uuid = db.Column(db.Text, primary_key=True, nullable=False)
    datetime = db.Column(db.DateTime, primary_key=True, nullable=False)
    site = db.Column(db.String(4), primary_key=True, nullable=False)
    factory = db.Column(db.String(4), primary_key=True, nullable=False)
    line = db.Column(db.String(8), nullable=False)
    aoi = db.Column(db.Text, nullable=False)
    product = db.Column(db.Text, nullable=False)
    board_count = db.Column(db.Integer)
    carrier_count = db.Column(db.Integer)



class AmrRawData(db.Model):
    __bind_key__ = 'amr'
    __tablename__ = 'amr_raw_data'
    __table_args__ = {'schema': 'public'}

    uuid = db.Column(db.Text, primary_key=True, nullable=False)
    product_name = db.Column(db.Text)
    site = db.Column(db.Text, primary_key=True, nullable=False)
    line_id = db.Column(db.Text)
    station_id = db.Column(db.Text)
    factory = db.Column(db.Text, primary_key=True, nullable=False)
    aoi_id = db.Column(db.Text)
    create_time = db.Column(db.DateTime, primary_key=True, nullable=False)
    top_btm = db.Column(db.Text)
    imulti_col = db.Column(db.Integer)
    imulti_row = db.Column(db.Integer)
    carrier_sn = db.Column(db.Text)
    board_sn = db.Column(db.Text)
    image_path = db.Column(db.Text)
    image_name = db.Column(db.Text)
    part_number = db.Column(db.Text)
    comp_name = db.Column(db.Text)
    window_id = db.Column(db.Integer)
    aoi_defect = db.Column(db.Text)
    op_defect = db.Column(db.Text)
    ai_result = db.Column(db.Text, info='1:OK(URD)   0:NG(IRI)')
    center_x = db.Column(db.Integer)
    center_y = db.Column(db.Integer)
    region_x = db.Column(db.Integer)
    region_y = db.Column(db.Integer)
    angle = db.Column(db.Float)
    cycle_time = db.Column(db.Float)
    total_comp = db.Column(db.Integer)
    package_type = db.Column(db.Text)
    comp_type = db.Column(db.Text)
    group_type = db.Column(db.Text)
    is_covered = db.Column(db.Boolean)
