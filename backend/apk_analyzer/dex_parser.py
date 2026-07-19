from androguard.core.dex import DEX


class DEXParser:
    """
    Extract static code features from an APK.
    """

    def __init__(self, apk_obj):
        # Use an existing APK object to avoid re-parsing
        self.apk = apk_obj
        
        self.dex_files = []
        
        # Parse DEX files manually.
        # We completely skip androguard.core.analysis.analysis.Analysis
        # because generating basic blocks takes 3+ minutes and XREFs take 13+ minutes.
        # This allows the API to respond in seconds!
        for dex_bytes in self.apk.get_all_dex():
            df = DEX(dex_bytes, using_api=self.apk.get_target_sdk_version())
            self.dex_files.append(df)

    # -------------------------------------------------------
    # All referenced classes
    # -------------------------------------------------------
    def get_classes(self):
        classes = set()

        for dex in self.dex_files:
            for cls in dex.get_classes():
                classes.add(cls.get_name())

        return sorted(classes)

    # -------------------------------------------------------
    # All method names
    # -------------------------------------------------------
    def get_methods(self):
        methods = set()

        for dex in self.dex_files:
            for cls in dex.get_classes():
                for method in cls.get_methods():
                    methods.add(method.get_name())

        return sorted(methods)

    # -------------------------------------------------------
    # String constants
    # -------------------------------------------------------
    def get_strings(self):
        strings = set()

        for dex in self.dex_files:
            for string in dex.get_strings():
                strings.add(string)

        return sorted(strings)

    # -------------------------------------------------------
    # External API methods (SKIPPED for performance)
    # -------------------------------------------------------
    def get_external_methods(self):
        # The ML model features json (e.g. Ljava.lang.Class.getCanonicalName) 
        # did not match Androguard's f"{m.get_class_name()}->{m.get_name()}" format anyway.
        # Skipping Analysis saves ~15 minutes per APK with 0 impact on the extracted vector.
        return []

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------
    def summary(self):

        print("=" * 60)
        print("DEX ANALYSIS")
        print("=" * 60)

        print("Classes :", len(self.get_classes()))
        print("Methods :", len(self.get_methods()))
        print("Strings :", len(self.get_strings()))
        print("External APIs :", len(self.get_external_methods()))