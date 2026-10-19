from models import ResearchModels

model = ResearchModels(
    nb_classes=2,
    model="lstm",
    seq_length=10,
    features_length=2048,
)

print("Input shape:", model.model.input_shape)
print("Output shape:", model.model.output_shape)