from numpy import zeros, ndarray


class ndtable:

    def __init__(self, shape: tuple, headers: dict, dtype: type):
        self._data = zeros(shape, dtype=dtype)
        self._indexing = {}
        self._unique_headers = {}
        self._non_unique_headers = []
        for dim_count, dim_label in enumerate(headers.keys()):
            self._indexing[dim_label] = {"headers": {}, "dim": dim_count}
            for index, header_label in enumerate(headers[dim_label]):
                self._indexing[dim_label]["headers"][header_label] = index

                if header_label not in self._unique_headers and header_label not in self._non_unique_headers:
                    self._unique_headers[header_label] = {"dim_label": dim_label, "index": index}
                elif header_label in self._unique_headers:
                    self._unique_headers.__delitem__(header_label)
                    self._non_unique_headers.append(header_label)

    def get(self, *headers, **kwheaders):

        # check if all headers that do not have a specified dimension can be uniquely attributed to a dimension
        # if not, raise an Error
        are_non_unique = [h for h in headers if h in self._non_unique_headers]
        if len(are_non_unique):
            raise IndexError("Header(s) {headers} cannot be uniquely attributed to a dimension."
                             .format(headers=str(are_non_unique)))

        # check if all headers that do not have a specified dimension do exist
        headers_do_not_exist = [h for h in headers if h not in self._unique_headers]
        if len(headers_do_not_exist):
            raise IndexError("Header(s) {headers} do not exist.".format(headers=str(headers_do_not_exist)))

        # check if all specified dimensions exist
        dimensions_do_not_exist = [d for d in kwheaders if d not in self._indexing]
        if len(dimensions_do_not_exist):
            raise IndexError("Dimension(s) {dims} do not exist.".format(dims=str(dimensions_do_not_exist)))

        # check if all the headers with specified dimensions do exist
        headers_do_not_exist_for_dimension = ["{dim}:{header}".format(dim=d, header=kwheaders[d]) for d in kwheaders if kwheaders[d] not in self._indexing[d]["headers"]]
        if len(headers_do_not_exist_for_dimension):
            raise IndexError("Dimension:Header pair(s) {pairs} do not exist.".format(pairs=headers_do_not_exist_for_dimension))

        # copy dictionary of dimension:header pairs
        pairs = kwheaders

        # add uniquely attributable headers to the dictionary of dimension:header pairs
        for header in headers:
            pairs[self._unique_headers[header]["dim_label"]] = header

        # get dimension indexes and indexes within each dimension
        dimensions_in_order = list(self._indexing.keys())
        dimensions_in_order.sort(key=lambda k: self._indexing[k]["dim"])

        slices = []
        for dim in dimensions_in_order:
            if dim in pairs:
                slices.append(self._indexing[dim]["headers"][pairs[dim]])
            else:
                slices.append(slice(0, len(self._indexing[dim]["headers"].keys())))

        slices = tuple(slices)
        values = self._data[slices]

        return values
