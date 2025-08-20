from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CapabilitiesResponse(_message.Message):
    __slots__ = ("framework_name", "framework_version", "epii_version", "supported_backends", "experiment_types", "extensions", "data_formats")
    class ExtensionsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    FRAMEWORK_NAME_FIELD_NUMBER: _ClassVar[int]
    FRAMEWORK_VERSION_FIELD_NUMBER: _ClassVar[int]
    EPII_VERSION_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_BACKENDS_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_TYPES_FIELD_NUMBER: _ClassVar[int]
    EXTENSIONS_FIELD_NUMBER: _ClassVar[int]
    DATA_FORMATS_FIELD_NUMBER: _ClassVar[int]
    framework_name: str
    framework_version: str
    epii_version: str
    supported_backends: _containers.RepeatedScalarFieldContainer[str]
    experiment_types: _containers.RepeatedCompositeFieldContainer[ExperimentSpec]
    extensions: _containers.ScalarMap[str, str]
    data_formats: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, framework_name: _Optional[str] = ..., framework_version: _Optional[str] = ..., epii_version: _Optional[str] = ..., supported_backends: _Optional[_Iterable[str]] = ..., experiment_types: _Optional[_Iterable[_Union[ExperimentSpec, _Mapping]]] = ..., extensions: _Optional[_Mapping[str, str]] = ..., data_formats: _Optional[_Iterable[str]] = ...) -> None: ...

class ExperimentSpec(_message.Message):
    __slots__ = ("name", "parameters", "output_parameters", "description")
    NAME_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    name: str
    parameters: _containers.RepeatedCompositeFieldContainer[ParameterSpec]
    output_parameters: _containers.RepeatedScalarFieldContainer[str]
    description: str
    def __init__(self, name: _Optional[str] = ..., parameters: _Optional[_Iterable[_Union[ParameterSpec, _Mapping]]] = ..., output_parameters: _Optional[_Iterable[str]] = ..., description: _Optional[str] = ...) -> None: ...

class ParameterSpec(_message.Message):
    __slots__ = ("name", "type", "required", "default_value", "description", "allowed_values")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    REQUIRED_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_VALUE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    ALLOWED_VALUES_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: str
    required: bool
    default_value: str
    description: str
    allowed_values: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, name: _Optional[str] = ..., type: _Optional[str] = ..., required: bool = ..., default_value: _Optional[str] = ..., description: _Optional[str] = ..., allowed_values: _Optional[_Iterable[str]] = ...) -> None: ...

class PingResponse(_message.Message):
    __slots__ = ("message", "timestamp")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    message: str
    timestamp: int
    def __init__(self, message: _Optional[str] = ..., timestamp: _Optional[int] = ...) -> None: ...

class ExperimentRequest(_message.Message):
    __slots__ = ("experiment_type", "parameters", "return_raw_data", "return_plots")
    class ParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    EXPERIMENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    RETURN_RAW_DATA_FIELD_NUMBER: _ClassVar[int]
    RETURN_PLOTS_FIELD_NUMBER: _ClassVar[int]
    experiment_type: str
    parameters: _containers.ScalarMap[str, str]
    return_raw_data: bool
    return_plots: bool
    def __init__(self, experiment_type: _Optional[str] = ..., parameters: _Optional[_Mapping[str, str]] = ..., return_raw_data: bool = ..., return_plots: bool = ...) -> None: ...

class ExperimentResponse(_message.Message):
    __slots__ = ("success", "error_message", "calibration_results", "measurement_data", "plots", "execution_time_seconds")
    class CalibrationResultsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CALIBRATION_RESULTS_FIELD_NUMBER: _ClassVar[int]
    MEASUREMENT_DATA_FIELD_NUMBER: _ClassVar[int]
    PLOTS_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_TIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error_message: str
    calibration_results: _containers.ScalarMap[str, float]
    measurement_data: _containers.RepeatedCompositeFieldContainer[NumpyArray]
    plots: _containers.RepeatedCompositeFieldContainer[PlotData]
    execution_time_seconds: float
    def __init__(self, success: bool = ..., error_message: _Optional[str] = ..., calibration_results: _Optional[_Mapping[str, float]] = ..., measurement_data: _Optional[_Iterable[_Union[NumpyArray, _Mapping]]] = ..., plots: _Optional[_Iterable[_Union[PlotData, _Mapping]]] = ..., execution_time_seconds: _Optional[float] = ...) -> None: ...

class NumpyArray(_message.Message):
    __slots__ = ("data", "shape", "dtype", "name", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    DATA_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    name: str
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, data: _Optional[bytes] = ..., shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., name: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class PlotData(_message.Message):
    __slots__ = ("plot_type", "title", "traces", "layout")
    class LayoutEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    PLOT_TYPE_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    TRACES_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_FIELD_NUMBER: _ClassVar[int]
    plot_type: str
    title: str
    traces: _containers.RepeatedCompositeFieldContainer[PlotTrace]
    layout: _containers.ScalarMap[str, str]
    def __init__(self, plot_type: _Optional[str] = ..., title: _Optional[str] = ..., traces: _Optional[_Iterable[_Union[PlotTrace, _Mapping]]] = ..., layout: _Optional[_Mapping[str, str]] = ...) -> None: ...

class PlotTrace(_message.Message):
    __slots__ = ("x", "y", "z", "name", "type")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    x: _containers.RepeatedScalarFieldContainer[float]
    y: _containers.RepeatedScalarFieldContainer[float]
    z: _containers.RepeatedScalarFieldContainer[float]
    name: str
    type: str
    def __init__(self, x: _Optional[_Iterable[float]] = ..., y: _Optional[_Iterable[float]] = ..., z: _Optional[_Iterable[float]] = ..., name: _Optional[str] = ..., type: _Optional[str] = ...) -> None: ...

class ExperimentsResponse(_message.Message):
    __slots__ = ("experiments",)
    EXPERIMENTS_FIELD_NUMBER: _ClassVar[int]
    experiments: _containers.RepeatedCompositeFieldContainer[ExperimentSpec]
    def __init__(self, experiments: _Optional[_Iterable[_Union[ExperimentSpec, _Mapping]]] = ...) -> None: ...

class ParametersListResponse(_message.Message):
    __slots__ = ("parameters",)
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    parameters: _containers.RepeatedCompositeFieldContainer[ParameterInfo]
    def __init__(self, parameters: _Optional[_Iterable[_Union[ParameterInfo, _Mapping]]] = ...) -> None: ...

class ParameterInfo(_message.Message):
    __slots__ = ("name", "type", "current_value", "description", "read_only")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    CURRENT_VALUE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    READ_ONLY_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: str
    current_value: str
    description: str
    read_only: bool
    def __init__(self, name: _Optional[str] = ..., type: _Optional[str] = ..., current_value: _Optional[str] = ..., description: _Optional[str] = ..., read_only: bool = ...) -> None: ...

class ParameterRequest(_message.Message):
    __slots__ = ("parameter_names",)
    PARAMETER_NAMES_FIELD_NUMBER: _ClassVar[int]
    parameter_names: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, parameter_names: _Optional[_Iterable[str]] = ...) -> None: ...

class ParametersResponse(_message.Message):
    __slots__ = ("parameters",)
    class ParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    parameters: _containers.ScalarMap[str, str]
    def __init__(self, parameters: _Optional[_Mapping[str, str]] = ...) -> None: ...

class SetParametersRequest(_message.Message):
    __slots__ = ("parameters",)
    class ParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    parameters: _containers.ScalarMap[str, str]
    def __init__(self, parameters: _Optional[_Mapping[str, str]] = ...) -> None: ...

class StatusResponse(_message.Message):
    __slots__ = ("success", "error_message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error_message: str
    def __init__(self, success: bool = ..., error_message: _Optional[str] = ...) -> None: ...
