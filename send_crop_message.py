import urllib.request
import json

FLASK_SERVER = "http://192.168.4.2:5000/crop_message"


def send_crop_message(message):

    data = {
        "message": message
    }

    json_data = json.dumps(data, ensure_ascii=False).encode("utf-8")

    request = urllib.request.Request(
        FLASK_SERVER,
        data=json_data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=5
        ) as response:

            response_data = response.read().decode("utf-8")

            print("✓ Crop result sent to Flask")
            print(response_data)

            return True

    except Exception as e:

        print("❌ Could not send crop result to Flask")
        print("Error:", e)

        return False