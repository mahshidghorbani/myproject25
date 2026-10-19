from extract_features import build_feature_extractor, extractor

model = build_feature_extractor()

print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

image_path = r"data\testing\Alert\Test_Alert1\z.jpg"

features = extractor(image_path, model)

print("Feature shape:", features.shape)
print("Feature dtype:", features.dtype)