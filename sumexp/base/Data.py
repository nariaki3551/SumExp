class Data(dict):
    """
    One Data

    Parameters
    ----------
    data_dict : dict
        key is item name, value if item value
    """
    def __init__(self, data_dict):
        super().__init__(data_dict)

    def __repr__(self):
        return f'Data({super().__repr__()})'
