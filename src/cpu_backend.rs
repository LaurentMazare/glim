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

impl<T: WithDType> crate::Backend<T> for Vec<T> {
    type Device = ();

    fn device(&self) -> &Self::Device {
        &()
    }

    // fn fill(&self, elem: T) -> Result<()> {
    //     self.fill(elem);
    //     Ok(())
    // }
    fn add_assign(&mut self, s: &Self) -> Result<()> {
        s.iter().zip(self.iter_mut()).for_each(|(src, dst)| *dst += *src);
        Ok(())
    }

    fn mul_assign(&mut self, s: &Self) -> Result<()> {
        s.iter().zip(self.iter_mut()).for_each(|(src, dst)| *dst *= *src);
        Ok(())
    }

    fn scale(&mut self, m: T) -> Result<()> {
        self.iter_mut().for_each(|v| *v *= m);
        Ok(())
    }
}

impl<T: WithDType + num_traits::Float> crate::BackendF<T> for Vec<T> {
    fn cos(&mut self) -> Result<()> {
        for d in self.iter_mut() {
            *d = d.cos();
        }
        Ok(())
    }

    fn sin(&mut self) -> Result<()> {
        for d in self.iter_mut() {
            *d = d.sin();
        }
        Ok(())
    }

    fn silu(&mut self) -> Result<()> {
        for d in self.iter_mut() {
            *d /= T::one() + (T::zero() - *d).exp()
        }
        Ok(())
    }
}
