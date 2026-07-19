import logging

# Hide Androguard debug logs
logging.getLogger("androguard").setLevel(logging.ERROR)

from androguard.core.apk import APK


class APKParser:
    def __init__(self, apk_path):
        self.apk = APK(apk_path)

    def get_basic_info(self):
        return {
            "App Name": self.apk.get_app_name(),
            "Package Name": self.apk.get_package(),
            "Version Name": self.apk.get_androidversion_name(),
            "Version Code": self.apk.get_androidversion_code(),
            "Min SDK": self.apk.get_min_sdk_version(),
            "Target SDK": self.apk.get_target_sdk_version(),
        }

    def get_permissions(self):
        return self.apk.get_permissions()

    def get_activities(self):
        return self.apk.get_activities()

    def get_services(self):
        return self.apk.get_services()

    def get_receivers(self):
        return self.apk.get_receivers()

    def get_providers(self):
        return self.apk.get_providers()

    def get_features(self):
        return self.apk.get_features()

    def get_libraries(self):
        return self.apk.get_libraries()