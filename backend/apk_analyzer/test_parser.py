import os
from apk_analyzer.apk_parser import APKParser

APK_PATH = "backend/uploads/F-Droid.apk"

print("=" * 60)
print("APK PARSER TEST")
print("=" * 60)

if not os.path.exists(APK_PATH):
    print("APK not found.")
    print("Expected:", APK_PATH)
    exit()

try:
    parser = APKParser(APK_PATH)

    print("\nBasic Information")
    print("-" * 60)

    info = parser.get_basic_info()

    for key, value in info.items():
        print(f"{key}: {value}")

    permissions = parser.get_permissions()
    activities = parser.get_activities()
    services = parser.get_services()
    receivers = parser.get_receivers()
    providers = parser.get_providers()

    print("\nPermissions:", len(permissions))
    for item in permissions:
        print(item)

    print("\nActivities:", len(activities))
    for item in activities:
        print(item)

    print("\nServices:", len(services))
    for item in services:
        print(item)

    print("\nReceivers:", len(receivers))
    for item in receivers:
        print(item)

    print("\nProviders:", len(providers))
    for item in providers:
        print(item)

    print("\n")
    print("=" * 60)
    print("APK Parsed Successfully")
    print("=" * 60)

except Exception as e:
    print("\nError")
    print(e)