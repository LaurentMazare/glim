use crate::Shape;

pub enum CowMut<'a, T> {
    Owned(T),
    Borrowed(&'a mut T),
}

impl<'a, T> std::ops::Deref for CowMut<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(o) => o,
            Self::Borrowed(r) => r,
        }
    }
}

impl<'a, T> std::ops::DerefMut for CowMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Owned(o) => o,
            Self::Borrowed(r) => r,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F16,
    BF16,
    F32,
}

pub trait WithDType: Sized + Copy + num_traits::Num {
    const DTYPE: DType;
}

impl WithDType for f32 {
    const DTYPE: DType = DType::F32;
}

#[derive(Clone)]
pub struct Storage<T: WithDType> {
    pub inner: Vec<T>,
}

pub struct Tensor<'a, T: WithDType> {
    data: CowMut<'a, Storage<T>>,
    shape: Shape,
}

impl<'a, T: WithDType> Tensor<'a, T> {
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    pub fn capacity(&self) -> usize {
        self.data.inner.len()
    }

    pub fn new<S: Into<Shape>>(data: &'a mut Storage<T>, shape: S) -> Self {
        let shape = shape.into();
        Self { data: CowMut::Borrowed(data), shape }
    }

    pub fn into_storage(self) -> CowMut<'a, Storage<T>> {
        self.data
    }
}

impl<T: WithDType> Tensor<'static, T> {
    pub fn zeros<S: Into<Shape>>(shape: S) -> Self {
        let shape = shape.into();
        let data = Storage { inner: vec![T::zero(); shape.elem_count()] };
        Self { data: CowMut::Owned(data), shape }
    }
}
