import json
import os

from apk_analyzer.apk_parser import APKParser
from apk_analyzer.dex_parser import DEXParser


class FeatureExtractor:

    def __init__(self, apk_parser, dex_parser):

        self.apk_parser = apk_parser
        self.dex_parser = dex_parser

        # Load feature list
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        features_path = os.path.join(project_root, "models", "drebin_features.json")
        with open(features_path, "r") as f:
            self.feature_names = json.load(f)

    # -------------------------------------------------------
    # Collect all extracted features
    # -------------------------------------------------------

    def collect_features(self):

        features = set()

        # Permissions
        for permission in self.apk_parser.get_permissions():
            features.add(permission.split(".")[-1])

        # Activities
        features.update(self.apk_parser.get_activities())

        # Services
        features.update(self.apk_parser.get_services())

        # Receivers
        features.update(self.apk_parser.get_receivers())

        # Methods
        features.update(self.dex_parser.get_methods())

        # External APIs
        features.update(self.dex_parser.get_external_methods())

        # Strings
        features.update(self.dex_parser.get_strings())

        # Classes
        features.update(self.dex_parser.get_classes())

        return features

    # -------------------------------------------------------
    # Generate 215 Feature Vector
    # -------------------------------------------------------

    def generate_vector(self):

        extracted = self.collect_features()

        vector = []

        for feature in self.feature_names:

            if feature in extracted:

                vector.append(1)

            else:

                vector.append(0)

        return vector