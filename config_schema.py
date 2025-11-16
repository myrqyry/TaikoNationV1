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
    effective_batch_size = fields.Int(validate=validate.Range(min=1, max=4096), required=True)
    num_epochs = fields.Int(validate=validate.Range(min=1, max=1000), required=True)
    k_folds = fields.Int(validate=validate.Range(min=2, max=20), required=True)
    use_wandb = fields.Bool(required=True)
    weight_decay = fields.Float(validate=validate.Range(min=0.0, max=0.5), required=True)
    warmup_ratio = fields.Float(validate=validate.Range(min=0.0, max=0.5), required=True)
    scheduler_type = fields.Str(validate=validate.OneOf(['cosine_warmup', 'linear']), required=True)
    restart_period = fields.Int(validate=validate.Range(min=0, max=100), required=True)
    early_stopping_patience = fields.Int(validate=validate.Range(min=0, max=100), required=True)
    early_stopping_delta = fields.Float(validate=validate.Range(min=0.0, max=0.1), required=True)
    num_workers = fields.Int(validate=validate.Range(min=0, max=32), required=True)
    prefetch_factor = fields.Int(validate=validate.Range(min=0, max=16), required=True)
    save_path = fields.Str(required=True)
    keep_best_n = fields.Int(validate=validate.Range(min=1, max=20), required=True)
    keep_last_n = fields.Int(validate=validate.Range(min=1, max=20), required=True)
    save_every_n_epochs = fields.Int(validate=validate.Range(min=0, max=100), required=True)
    cleanup_checkpoints = fields.Bool(required=True)

class GenerateConfigSchema(Schema):
    max_output_length = fields.Int(validate=validate.Range(min=64, max=8192), required=True)
    default_difficulty = fields.Str(validate=validate.OneOf(['easy', 'normal', 'hard', 'oni', 'ura']), required=True)
    beam_size = fields.Int(validate=validate.Range(min=1, max=16), required=True)

class ConfigSchema(Schema):
    model = fields.Nested(ModelConfigSchema, required=True)
    training = fields.Nested(TrainingConfigSchema, required=True)
    data = fields.Nested(DataConfigSchema, required=True)
    generate = fields.Nested(GenerateConfigSchema, required=True)
