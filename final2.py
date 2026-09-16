import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import json
from send_crop_message import send_crop_message

# ============================================================
# DATA FOR OTHER FILES
# ============================================================

report_lines = []

farm_report = {
    "leaves": 0,
    "flowers": 0,
    "fruits": 0,
    "ripe_fruits": 0,
    "raw_fruits": 0,
    "diseases": [],
    "leaf_results": [],
    "terminal_messages": report_lines
}


# ============================================================
# FUNCTION TO PRINT + SAVE EVERY MESSAGE
# ============================================================

def show(message):
    print(message)
    report_lines.append(str(message))


# ============================================================
# SAVE REPORT FOR OTHER PYTHON FILES
# ============================================================

def save_report():

    with open(
        "farm_report.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            farm_report,
            file,
            ensure_ascii=False,
            indent=4
        )


# ============================================================
# HINDI FARMER-FRIENDLY LANGUAGE
# ============================================================

DISEASE_HINDI = {

    "Tomato___Early_blight":
        "टमाटर में अर्ली ब्लाइट बीमारी",

    "Tomato___Late_blight":
        "टमाटर में लेट ब्लाइट बीमारी",

    "Tomato___healthy":
        "टमाटर का पौधा स्वस्थ है",

    "Potato___Early_blight":
        "आलू में अर्ली ब्लाइट बीमारी",

    "Potato___Late_blight":
        "आलू में लेट ब्लाइट बीमारी",

    "Potato___healthy":
        "आलू का पौधा स्वस्थ है",

    "Pepper_bell___Bacterial_spot":
        "शिमला मिर्च में बैक्टीरियल स्पॉट बीमारी",

    "Pepper_bell___healthy":
        "शिमला मिर्च का पौधा स्वस्थ है",
}


def hindi_disease_name(disease):

    return DISEASE_HINDI.get(
        disease,
        disease.replace("___", " ").replace("_", " ")
    )


def hindi_ripeness(ripeness):

    name = ripeness.lower()

    if "unripe" in name or "raw" in name:
        return "कच्चा फल"

    if "ripe" in name:
        return "पका हुआ फल"

    return ripeness


# ============================================================
# HINGLISH FARMER-FRIENDLY LANGUAGE
# ============================================================

def hinglish_disease_name(disease):

    HINGLISH_DISEASE = {

        "Tomato___Early_blight":
            "Patti mein Early Blight bimari",

        "Tomato___Late_blight":
            "Patti mein Late Blight bimari",

        "Tomato___healthy":
            "Patti ka paudha healthy hai",

        "Potato___Early_blight":
            "Patti mein Early Blight bimari",

        "Potato___Late_blight":
            "Patti mein Late Blight bimari",

        "Potato___healthy":
            "Patti ka paudha healthy hai",

        "Pepper_bell___Bacterial_spot":
            "Patti mein Bacterial Spot bimari",

        "Pepper_bell___healthy":
            "Patti ka paudha healthy hai",
    }

    return HINGLISH_DISEASE.get(
        disease,
        disease.replace("___", " ").replace("_", " ")
    )


def hinglish_ripeness(ripeness):

    name = ripeness.lower()

    if "unripe" in name or "raw" in name:
        return "Kaccha phal"

    if "ripe" in name:
        return "Paka hua phal"

    return ripeness


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

show(f"Device: {device}")


# ============================================================
# 2. LOAD YOLO MODEL
# ============================================================

yolo_model = YOLO("sam_model.pt")

show("✓ YOLO model loaded")


# ============================================================
# 3. LOAD RIPENESS MODEL
# ============================================================

# Ripeness model currently disabled.

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

# show("✓ Ripeness model loaded")
# show(f"Ripeness classes: {len(ripeness_classes)}")


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

show("✓ Disease model loaded")
show(f"Disease classes: {len(disease_classes)}")


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

    # Ripeness model currently disabled.


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

    show("❌ Camera not detected")

    exit()


show("")
show("=" * 60)
show("🌱 FARM AI")
show("=" * 60)
show("Press S = capture and analyze")
show("Press Q = quit")
show("=" * 60)


# ============================================================
# 9. CAMERA LOOP
# ============================================================

while True:

    success, frame = camera.read()

    if not success:

        show("❌ Could not read camera")

        break


    cv2.imshow(
        "Farm AI Camera",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    # ========================================================
    # CAPTURE
    # ========================================================

    if key == ord("s"):

        show("")
        show("=" * 60)
        show("📸 ANALYZING FARM IMAGE")
        show("=" * 60)


        # ----------------------------------------------------
        # RESET REPORT DATA FOR THIS CAPTURE
        # ----------------------------------------------------

        farm_report["leaves"] = 0
        farm_report["flowers"] = 0
        farm_report["fruits"] = 0
        farm_report["ripe_fruits"] = 0
        farm_report["raw_fruits"] = 0
        farm_report["diseases"] = []
        farm_report["leaf_results"] = []
        report_lines.clear()


        # ----------------------------------------------------
        # RUN YOLO
        # ----------------------------------------------------

        results = yolo_model(frame)

        result = results[0]


        # ----------------------------------------------------
        # COUNTERS
        # ----------------------------------------------------

        leaf_count = 0
        flower_count = 0
        fruit_count = 0

        ripe_count = 0
        raw_count = 0


        # ====================================================
        # PROCESS EVERY DETECTED OBJECT
        # ====================================================

        for i in range(len(result.boxes)):

            # ------------------------------------------------
            # CLASS ID
            # ------------------------------------------------

            class_id = int(
                result.boxes.cls[i]
            )


            # ------------------------------------------------
            # CLASS NAME
            # ------------------------------------------------

            class_name = result.names[class_id]


            # ------------------------------------------------
            # DETECTION CONFIDENCE
            # ------------------------------------------------

            detection_confidence = float(
                result.boxes.conf[i]
            ) * 100


            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            x1, y1, x2, y2 = (
                result.boxes.xyxy[i]
                .int()
                .tolist()
            )


            # ------------------------------------------------
            # KEEP COORDINATES INSIDE IMAGE
            # ------------------------------------------------

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(
                frame.shape[1],
                x2
            )

            y2 = min(
                frame.shape[0],
                y2
            )


            # ------------------------------------------------
            # CROP DETECTED OBJECT
            # ------------------------------------------------

            crop = frame[
                y1:y2,
                x1:x2
            ]


            if crop.size == 0:

                continue


            # =================================================
            # LEAF
            # =================================================

            if class_name.lower() == "leaf":

                leaf_count += 1


                # ---------------------------------------------
                # DISEASE MODEL
                # ---------------------------------------------

                disease, disease_confidence = (
                    predict_disease(crop)
                )
                send_crop_message(
                    f"{hinglish_disease_name(disease)} "
                    f"({disease_confidence:.2f}% confidence)"
                )

                # ---------------------------------------------
                # SAVE DISEASE DATA
                # ---------------------------------------------

                disease_data = {

                    "disease": disease,

                    "hindi":
                        hindi_disease_name(disease),

                    "hinglish":
                        hinglish_disease_name(disease),

                    "confidence":
                        disease_confidence,

                    "leaf_detection_confidence":
                        detection_confidence
                }


                farm_report[
                    "diseases"
                ].append(disease_data)


                farm_report[
                    "leaf_results"
                ].append(disease_data)


                # ---------------------------------------------
                # HINDI OUTPUT
                # ---------------------------------------------

                show("\n🍃 पत्ती की जाँच")

                show(
                    f"पत्ती मिलने का भरोसा: "
                    f"{detection_confidence:.2f}%"
                )

                show(
                    f"समस्या: "
                    f"{hindi_disease_name(disease)}"
                )

                show(
                    f"जाँच का भरोसा: "
                    f"{disease_confidence:.2f}%"
                )


                if "healthy" in disease.lower():

                    show(
                        "✅ यह पत्ती स्वस्थ दिखाई दे रही है।"
                    )

                else:

                    show(
                        "⚠️ इस पत्ती में बीमारी के "
                        "लक्षण दिखाई दे रहे हैं।"
                    )


                # ---------------------------------------------
                # HINGLISH OUTPUT
                # ---------------------------------------------

                show("\n🍃 LEAF CHECK - HINGLISH")

                show(
                    f"Leaf detection confidence: "
                    f"{detection_confidence:.2f}%"
                )

                show(
                    f"Problem: "
                    f"{hinglish_disease_name(disease)}"
                )

                show(
                    f"Disease check confidence: "
                    f"{disease_confidence:.2f}%"
                )


                if "healthy" in disease.lower():

                    show(
                        "✅ Yeh leaf healthy dikh rahi hai."
                    )

                else:

                    show(
                        "⚠️ Is leaf mein disease ke "
                        "symptoms dikh rahe hain."
                    )


            # =================================================
            # FLOWER
            # =================================================

            elif class_name.lower() == "flower":

                flower_count += 1


                # ---------------------------------------------
                # HINDI
                # ---------------------------------------------

                show("\n🌸 फूल की जाँच")

                show(
                    f"फूल मिलने का भरोसा: "
                    f"{detection_confidence:.2f}%"
                )


                # ---------------------------------------------
                # HINGLISH
                # ---------------------------------------------

                show("\n🌸 FLOWER CHECK - HINGLISH")

                show(
                    f"Flower detection confidence: "
                    f"{detection_confidence:.2f}%"
                )


            # =================================================
            # FRUIT
            # =================================================

            elif class_name.lower() == "fruit":

                fruit_count += 1


                # ---------------------------------------------
                # HINGLISH
                # ---------------------------------------------

                show("\n🍅 FRUIT CHECK - HINGLISH")

                show(
                    f"Fruit detection confidence: "
                    f"{detection_confidence:.2f}%"
                )


                # ------------------------------------------------
                # RIPENESS MODEL CURRENTLY DISABLED
                # ------------------------------------------------

                # ripeness, ripeness_confidence = (
                #     predict_ripeness(crop)
                # )

                # show("\n🍅 फल की जाँच")

                # show(
                #     f"फल मिलने का भरोसा: "
                #     f"{detection_confidence:.2f}%"
                # )

                # show(
                #     f"फल की स्थिति: "
                #     f"{hindi_ripeness(ripeness)}"
                # )

                # show(
                #     f"जाँच का भरोसा: "
                #     f"{ripeness_confidence:.2f}%"
                # )

                # if "unripe" in ripeness.lower() or \
                #    "raw" in ripeness.lower():

                #     show(
                #         "🟡 यह फल अभी कच्चा है।"
                #     )

                #     raw_count += 1

                # elif "ripe" in ripeness.lower():

                #     show(
                #         "🟢 यह फल पक चुका है।"
                #     )

                #     ripe_count += 1


        # =====================================================
        # UPDATE MAIN REPORT VARIABLES
        # =====================================================

        farm_report["leaves"] = leaf_count

        farm_report["flowers"] = flower_count

        farm_report["fruits"] = fruit_count

        farm_report["ripe_fruits"] = ripe_count

        farm_report["raw_fruits"] = raw_count


        # =====================================================
        # FINAL FARM IMAGE REPORT - HINDI
        # =====================================================

        show("\n")
        show("=" * 60)
        show("🌾 खेत की जाँच रिपोर्ट")
        show("=" * 60)


        show(
            f"🍃 कुल पत्तियाँ: {leaf_count}"
        )

        show(
            f"🌸 कुल फूल: {flower_count}"
        )

        show(
            f"🍅 कुल फल: {fruit_count}"
        )

        show(
            f"🟢 पके हुए फल: {ripe_count}"
        )

        show(
            f"🟡 कच्चे फल: {raw_count}"
        )


        show("\n📋 आसान जानकारी:")

        show(
            f"आपकी तस्वीर में कुल {leaf_count} पत्तियाँ, "
            f"{flower_count} फूल और {fruit_count} फल मिले।"
        )


        if ripe_count > 0:

            show(
                f"🍅 इनमें से {ripe_count} फल पक चुके हैं।"
            )


        if raw_count > 0:

            show(
                f"🍅 {raw_count} फल अभी कच्चे हैं।"
            )


        show("=" * 60)


        # =====================================================
        # HINGLISH FARM IMAGE REPORT
        # =====================================================

        show("\n")
        show("=" * 60)
        show("🌾 FARM AI REPORT - HINGLISH")
        show("=" * 60)


        show(
            f"🍃 Total leaves: {leaf_count}"
        )

        show(
            f"🌸 Total flowers: {flower_count}"
        )

        show(
            f"🍅 Total fruits: {fruit_count}"
        )

        show(
            f"🟢 Ripe fruits: {ripe_count}"
        )

        show(
            f"🟡 Raw fruits: {raw_count}"
        )


        show("\n📋 Easy Information:")


        show(
            f"Aapki khet mein total {leaf_count} leaves, "
            f"{flower_count} flowers aur {fruit_count} fruits mile."
        )


        if ripe_count > 0:

            show(
                f"🍅 Inmein se {ripe_count} fruits pak chuke hain."
            )


        if raw_count > 0:

            show(
                f"🍅 {raw_count} fruits abhi kacche hain."
            )


        show("=" * 60)


        # =====================================================
        # SAVE COMPLETE REPORT
        # =====================================================

        save_report()

        show("✓ Farm report saved to farm_report.json")


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