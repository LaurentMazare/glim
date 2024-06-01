use crate::WithDType;
use anyhow::Result;

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

#[derive(Clone)]
pub struct Storage<T: WithDType> {
    pub inner: Vec<T>,
}

impl<T: WithDType> Storage<T> {
    pub fn cst(t: T, elts: usize) -> Result<Self> {
        Ok(Self { inner: vec![t; elts] })
    }
}
