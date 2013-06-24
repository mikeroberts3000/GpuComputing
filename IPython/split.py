import math
import numpy
import pycuda.autoinit
import pycuda.driver
import pycuda.compiler
import prefixsum

class SplitManager:
    
    source_module = pycuda.compiler.SourceModule \
    (
    """
    __global__ void split_scatter(
        unsigned int* d_input_data,
        unsigned int* d_flag_data,
        unsigned int* d_flag_set_scatter_offset,
        unsigned int* d_output_data,
        int total_flags_set,
        int n )
    {
        int global_index_1d = ( blockIdx.x * blockDim.x ) + threadIdx.x;

        if ( global_index_1d < n )
        {
            unsigned int input_value                   = d_input_data[ global_index_1d ];
            unsigned int flag_value                    = d_flag_data[ global_index_1d ];
            unsigned int flag_set_scatter_offset_value = d_flag_set_scatter_offset[ global_index_1d ];
            
            unsigned int scatter_offset_value;

            if ( flag_value > 0 )
            {
                scatter_offset_value = flag_set_scatter_offset_value;
            }
            else
            {
                scatter_offset_value = global_index_1d - flag_set_scatter_offset_value + total_flags_set;
            }
        
            d_output_data[ scatter_offset_value ] = input_value;
        }
    }
    """
    )

    _split_scatter_funcion = source_module.get_function("split_scatter")

    _prefix_sum_manager              = -1

    _size_of_element_bytes           = 4
    _max_num_elements                = -1
    _num_bytes                       = -1
    _n                               = -1
    _input_data_device               = -1
    _flag_data_device                = -1
    _flag_set_scatter_offset_device  = -1
    _output_data_device              = -1
    _block_sums_device               = -1
    
    def __init__(self, max_num_elements):
        
        self._max_num_elements               = max_num_elements
        self._num_bytes                      = self._max_num_elements * self._size_of_element_bytes
        self._input_data_device              = pycuda.driver.mem_alloc(self._num_bytes)
        self._flag_data_device               = pycuda.driver.mem_alloc(self._num_bytes)
        self._flag_set_scatter_offset_device = pycuda.driver.mem_alloc(self._num_bytes)
        self._output_data_device             = pycuda.driver.mem_alloc(self._num_bytes)
        self._prefix_sum_manager             = prefixsum.PrefixSumManager(self._max_num_elements)

    def __copy_input_htod(self, input_data_host, flag_data_host):

        assert input_data_host.shape[0] <= self._max_num_elements
        assert \
            input_data_host.dtype == numpy.uint32  or \
            input_data_host.dtype == numpy.int32   or \
            input_data_host.dtype == numpy.float32

        pycuda.driver.memcpy_htod(self._input_data_device, input_data_host)
        pycuda.driver.memcpy_htod(self._flag_data_device,  flag_data_host)
        
    def __split(self, input_data_device, flag_data_device, output_data_device, num_elements):

        assert num_elements <= self._max_num_elements

        self._n = num_elements
        
        pycuda.driver.memset_d32(self._flag_set_scatter_offset_device, 0, self._n)
        pycuda.driver.memset_d32(output_data_device,                   0, self._n)

        self._prefix_sum_manager.prefix_sum_device(flag_data_device, self._flag_set_scatter_offset_device, self._n)

        tmp = numpy.zeros(1, dtype=numpy.uint32)

        pycuda.driver.memcpy_dtoh(tmp, int(self._flag_set_scatter_offset_device) + ((self._n - 1) * self._size_of_element_bytes))
        flag_set_scatter_offset_end = tmp[0]

        pycuda.driver.memcpy_dtoh(tmp, int(flag_data_device) + ((self._n - 1) * self._size_of_element_bytes))
        flag_data_end = tmp[0]

        total_flags_set = flag_set_scatter_offset_end + flag_data_end

        split_scatter_funcion_block = (512,1,1)
        num_blocks                  = int(math.ceil(float(self._n) / float(split_scatter_funcion_block[0])))
        split_scatter_function_grid = (num_blocks, 1)

        self._split_scatter_funcion(
            input_data_device,
            flag_data_device,
            self._flag_set_scatter_offset_device,
            output_data_device,
            numpy.int32(total_flags_set),
            numpy.int32(self._n),
            block=split_scatter_funcion_block,
            grid=split_scatter_function_grid)
        
    def __copy_output_dtoh(self, output_data_host):

        pycuda.driver.memcpy_dtoh(output_data_host, self._output_data_device)

    def split_device(self, input_data_device, flag_data_device, output_data_device, num_elements):

        self.__split(input_data_device, flag_data_device, output_data_device, num_elements)

    def split_host(self, input_data_host, flag_data_host, output_data_host):

        num_elements = input_data_host.shape[0]
        
        self.__copy_input_htod(input_data_host, flag_data_host)
        self.split_device(self._input_data_device, self._flag_data_device, self._output_data_device, num_elements)
        self.__copy_output_dtoh(output_data_host)
