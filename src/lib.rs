pub mod device;
pub mod dtype;
pub mod kv_cache;
pub mod llama;
pub mod shape;
pub mod storage;
pub mod tensor;
pub mod tensor_view;

pub use dtype::{DType, WithDType};
pub use shape::{Dim, Shape, D};
pub use tensor::Tensor;
pub use tensor_view::{TensorOrView, TensorView};

pub type TensorS<T> = Tensor<'static, T>;
