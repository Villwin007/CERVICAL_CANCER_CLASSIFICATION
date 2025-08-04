Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# Example placeholder (to be filled post search)
DARTS_V1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('skip_connect', 0), ('sep_conv_3x3', 1),
            ('skip_connect', 0), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('dil_conv_3x3', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1),
            ('skip_connect', 2), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('skip_connect', 2)],
    reduce_concat=[2, 3, 4, 5],
)
