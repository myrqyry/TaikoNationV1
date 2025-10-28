from marshmallow import Schema, fields, validate

class ModelConfigSchema(Schema):
    d_model = fields.Int(validate=validate.Range(min=64, max=2048), required=True)
    nhead = fields.Int(validate=validate.Range(min=1, max=32), required=True)
    num_encoder_layers = fields.Int(validate=validate.Range(min=1, max=24), required=True)
    num_decoder_layers = fields.Int(validate=validate.Range(min=1, max=24), required=True)
    dim_feedforward = fields.Int(validate=validate.Range(min=256, max=8192), required=True)
    dropout = fields.Float(validate=validate.Range(min=0.0, max=0.9), required=True)
    audio_feature_size = fields.Int(validate=validate.Range(min=1, max=1024), required=True)

class DataConfigSchema(Schema):
    max_sequence_length = fields.Int(validate=validate.Range(min=64, max=4096), required=True)
    time_quantization_ms = fields.Int(validate=validate.Range(min=1, max=1000), required=True)
    source_resolution_ms = fields.Float(validate=validate.Range(min=1.0, max=1000.0), required=True)

class TrainingConfigSchema(Schema):
    learning_rate = fields.Float(validate=validate.Range(min=1e-7, max=1e-1), required=True)
    batch_size = fields.Int(validate=validate.Range(min=1, max=1024), required=True)
    num_epochs = fields.Int(validate=validate.Range(min=1, max=1000), required=True)
    save_path = fields.Str(required=True)
    k_folds = fields.Int(validate=validate.Range(min=2, max=20), required=True)

class ConfigSchema(Schema):
    model = fields.Nested(ModelConfigSchema, required=True)
    training = fields.Nested(TrainingConfigSchema, required=True)
    data = fields.Nested(DataConfigSchema, required=True)
