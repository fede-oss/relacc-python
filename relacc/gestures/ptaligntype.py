class PtAlignType:
    CHRONOLOGICAL = 0
    CLOUD_MATCH = 1

    @classmethod
    def normalize(cls, value):
        if isinstance(value, bool):
            raise ValueError(cls._error(value))
        if isinstance(value, str):
            value = value.strip().lower()
            aliases = {
                "0": cls.CHRONOLOGICAL,
                "chronological": cls.CHRONOLOGICAL,
                "1": cls.CLOUD_MATCH,
                "cloud": cls.CLOUD_MATCH,
                "cloud-match": cls.CLOUD_MATCH,
            }
            if value in aliases:
                return aliases[value]
            raise ValueError(cls._error(value))
        if type(value) is int and value in (cls.CHRONOLOGICAL, cls.CLOUD_MATCH):
            return value
        raise ValueError(cls._error(value))

    @classmethod
    def name(cls, value):
        normalized = cls.normalize(value)
        if normalized == cls.CHRONOLOGICAL:
            return "chronological"
        return "cloud-match"

    @staticmethod
    def _error(value):
        return (
            "Invalid alignment (%r). Supported values: 0, chronological, "
            "1, cloud, cloud-match." % value
        )
