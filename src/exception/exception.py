class VmatchException(Exception):
    """vmatch exception"""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"Vmatch exception: {self.message}"

    def __repr__(self):
        return f"Vmatch exception: {self.message}"
