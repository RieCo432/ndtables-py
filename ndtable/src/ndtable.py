from numpy import zeros


class ndtable:

    def __init__(self, shape: tuple, headers: dict, dtype: type):
        self.data = zeros(shape, dtype=dtype)
        self.indexing = {}
        self._unique_headers = {}
        self._non_unique_headers = []
        for dim_count, dim_label in enumerate(headers.keys()):
            self.indexing[dim_label] = {"headers": {}, "dim": dim_count}
            for index, header_label in enumerate(headers[dim_label]):
                self.indexing[dim_label]["headers"][header_label] = index

                if header_label not in self._unique_headers and header_label not in self._non_unique_headers:
                    self._unique_headers[header_label] = {"dim": dim_count, "index": index}
                elif header_label in self._unique_headers:
                    self._unique_headers.__delitem__(header_label)
                    self._non_unique_headers.append(header_label)

    def __getitem__(self, *headers):
        pass
