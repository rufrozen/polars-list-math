"""Module that must fail during import because its model has an unknown type."""

import polars_list_math.typed_polars2 as tp


class UnsupportedValue:
    pass


@tp.model
class InvalidModel(tp.Model):
    value: UnsupportedValue
