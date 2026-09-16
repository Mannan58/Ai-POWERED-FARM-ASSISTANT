import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image

# ============================================================
# HINDI FARMER-FRIENDLY LANGUAGE
# ============================================================

DISEASE_HINDI = {
    "Tomato___Early_blight": "टमाटर में अर्ली ब्लाइट बीमारी",
    "Tomato___Late_blight": "टमाटर में लेट ब्लाइट बीमारी",
    "Tomato___healthy": "टमाटर का पौधा स्वस्थ है",

    "Potato___Early_blight": "आलू में अर्ली ब्लाइट बीमारी",
    "Potato___Late_blight": "आलू में लेट ब्लाइट बीमारी",
    "Potato___healthy": "आलू का पौधा स्वस्थ है",

    "Pepper_bell___Bacterial_spot": "शिमला मिर्च में बैक्टीरियल स्पॉट बीमारी",
    "Pepper_bell___healthy": "शिमला मिर्च का पौधा स्वस्थ है",
}


def hindi_disease_name(disease):
    return DISEASE_HINDI.get(
        disease,
        disease.replace("___", " ").replace("_", " ")
    )


def hindi_ripeness(ripeness):

    name = ripeness.lower()

    if "ripe" in name:
        return "पका हुआ फल"

    if "raw" in name or "unripe" in name:
        return "कच्चा फल"

    return ripeness

# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# 2. LOAD YOLO MODEL
# ============================================================

yolo_model = YOLO("sam_model.pt")

print("✓ YOLO model loaded")


# ============================================================
# 3. LOAD RIPENESS MODEL
# ============================================================

# ripeness_checkpoint = torch.load(
#     "rawripe_ripeness_model.pth",
#     map_location=device
# )

# ripeness_classes = ripeness_checkpoint["classes"]

# ripeness_model = models.resnet18(
#     weights=None
# )

# ripeness_model.fc = nn.Linear(
#     ripeness_model.fc.in_features,
#     len(ripeness_classes)
# )

# ripeness_model.load_state_dict(
#     ripeness_checkpoint["model_state_dict"]
# )

# ripeness_model = ripeness_model.to(device)

# ripeness_model.eval()

# print("✓ Ripeness model loaded")
# print("Ripeness classes:", len(ripeness_classes))


# ============================================================
# 4. LOAD DISEASE MODEL
# ============================================================

disease_checkpoint = torch.load(
    "plant_disease_model_best.pth",
    map_location=device
)

disease_classes = disease_checkpoint["classes"]

disease_model = models.resnet18(
    weights=None
)

disease_model.fc = nn.Linear(
    disease_model.fc.in_features,
    len(disease_classes)
)

disease_model.load_state_dict(
    disease_checkpoint["model_state_dict"]
)

disease_model = disease_model.to(device)

disease_model.eval()

print("✓ Disease model loaded")
print("Disease classes:", len(disease_classes))


# ============================================================
# 5. IMAGE PREPROCESSING
# ============================================================

image_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# 6. RIPENESS PREDICTION
# ============================================================

def predict_ripeness(crop):

    rgb_crop = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(rgb_crop)

    image = image_transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    # with torch.no_grad():

        # output = ripeness_model(image)

        # probabilities = torch.softmax(
        #     output,
        #     dim=1
        # )

        # confidence, prediction = torch.max(
        #     probabilities,
        #     dim=1
        # )

    # class_name = ripeness_classes[
    #     prediction.item()
    # ]

    # confidence = confidence.item() * 100

    # return class_name, confidence


# ============================================================
# 7. DISEASE PREDICTION
# ============================================================

def predict_disease(crop):

    rgb_crop = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(rgb_crop)

    image = image_transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        output = disease_model(image)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    class_name = disease_classes[
        prediction.item()
    ]

    confidence = confidence.item() * 100

    return class_name, confidence


# ============================================================
# 8. CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("❌ Camera not detected")

    exit()


print("\n")
print("=" * 60)
print("🌱 FARM AI")
print("=" * 60)
print("Press S = capture and analyze")
print("Press Q = quit")
print("=" * 60)


# ============================================================
# 9. CAMERA LOOP
# ============================================================

while True:

    success, frame = camera.read()

    if not success:

        print("❌ Could not read camera")

        break


    cv2.imshow(
        "Farm AI Camera",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # CAPTURE
    # --------------------------------------------------------

    if key == ord("s"):

        print("\n")
        print("=" * 60)
        print("📸 ANALYZING FARM IMAGE")
        print("=" * 60)


        # Run YOLO

        results = yolo_model(frame)

        result = results[0]


        # Counters

        leaf_count = 0
        flower_count = 0
        fruit_count = 0

        ripe_count = 0
        raw_count = 0


        # ----------------------------------------------------
        # PROCESS EVERY DETECTED OBJECT
        # ----------------------------------------------------

        for i in range(len(result.boxes)):

            # Class ID
            class_id = int(result.boxes.cls[i])

            # Class name
            class_name = result.names[class_id]

            # Detection confidence
            detection_confidence = float(
                result.boxes.conf[i]
            ) * 100

            # Bounding box
            x1, y1, x2, y2 = (
                result.boxes.xyxy[i]
                .int()
                .tolist()
            )

            # Keep coordinates inside image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            # Crop detected object
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # =================================================
            # LEAF
            # =================================================

            if class_name.lower() == "leaf":

                leaf_count += 1

                disease, disease_confidence = (
                    predict_disease(crop)
                )

                print("\n🍃 पत्ती की जाँच")

                print(
                    f"पत्ती मिलने का भरोसा: "
                    f"{detection_confidence:.2f}%"
                )

                print(
                    f"समस्या: "
                    f"{hindi_disease_name(disease)}"
                )

                print(
                    f"जाँच का भरोसा: "
                    f"{disease_confidence:.2f}%"
                )

                if "healthy" in disease.lower():

                    print(
                        "✅ यह पत्ती स्वस्थ दिखाई दे रही है।"
                    )

                else:

                    print(
                        "⚠️ इस पत्ती में बीमारी के "
                        "लक्षण दिखाई दे रहे हैं।"
                    )

            # =================================================
            # FLOWER
            # =================================================

            elif class_name.lower() == "flower":

                flower_count += 1

                print("\n🌸 फूल की जाँच")

                print(
                    f"फूल मिलने का भरोसा: "
                    f"{detection_confidence:.2f}%"
                )

            # =================================================
            # FRUIT
            # =================================================

            elif class_name.lower() == "fruit":

                fruit_count += 1

                # ripeness, ripeness_confidence = (
                #     predict_ripeness(crop)
                # )

                # print("\n🍅 फल की जाँच")

                # print(
                #     f"फल मिलने का भरोसा: "
                #     f"{detection_confidence:.2f}%"
                # )

                # print(
                #     f"फल की स्थिति: "
                #     f"{hindi_ripeness(ripeness)}"
                # )

                # print(
                #     f"जाँच का भरोसा: "
                #     f"{ripeness_confidence:.2f}%"
                # )

                # if "ripe" in ripeness.lower():

                #     print(
                #         "🟢 यह फल पक चुका है।"
                #     )

                #     ripe_count += 1

                # elif "raw" in ripeness.lower():

                #     print(
                #         "🟡 यह फल अभी कच्चा है।"
                #     )

                #     raw_count += 1        # =====================================================
        # FINAL FARM IMAGE REPORT
        # =====================================================

        print("\n")
        print("=" * 60)
        print("\n")
        print("=" * 60)
        print("🌾 खेत की जाँच रिपोर्ट")
        print("=" * 60)

        print(
            f"🍃 कुल पत्तियाँ: {leaf_count}"
        )

        print(
            f"🌸 कुल फूल: {flower_count}"
        )

        print(
            f"🍅 कुल फल: {fruit_count}"
        )

        print(
            f"🟢 पके हुए फल: {ripe_count}"
        )

        print(
            f"🟡 कच्चे फल: {raw_count}"
        )

        print("\n📋 आसान जानकारी:")

        print(
            f"आपकी तस्वीर में कुल {leaf_count} पत्तियाँ, "
            f"{flower_count} फूल और {fruit_count} फल मिले।"
        )

        if ripe_count > 0:

            print(
                f"🍅 इनमें से {ripe_count} फल पक चुके हैं।"
            )

        if raw_count > 0:

            print(
                f"🍅 {raw_count} फल अभी कच्चे हैं।"
            )

        print("=" * 60)
        print("=" * 60)


    # ========================================================
    # QUIT
    # ========================================================

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()
