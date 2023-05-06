use std::cmp::Ordering;
use std::fmt::{Display, Formatter};
use std::ops::Index;
use num_traits::{Float, Num, NumOps};

macro_rules! vector {
    ($($x:expr),*) => {
        Vector::new(vec![$($x as f64),*])
    };
}

#[derive(Clone, Debug)]
pub struct Vector<T> {
    components: Vec<T>,
}

impl<T: std::str::FromStr> TryFrom<String> for Vector<T> {
    type Error = &'static str;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        let components_str = s
            .trim_matches(|c| c == '[' || c == ']')
            .split(',')
            .map(str::trim);

        let components: Result<Vec<T>, _> = components_str.map(str::parse).collect();
        match components {
            Ok(components) => Ok(Self { components }),
            Err(_) => Err("could not parse component"),
        }
    }
}

impl<T: PartialEq> PartialEq for Vector<T> {
    /// Compares this vector to another for equality. Two vectors are considered equal if they have
    /// the same dimension and their corresponding components are equal. Returns `true` if the
    /// vectors are equal and `false` otherwise.
    ///
    /// # Arguments
    ///
    /// * `other` - The other vector to compare to.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_linear_algebra::math::Vector;
    ///
    /// let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
    /// let v2 = Vector::new(vec![1.0, 2.0, 3.0]);
    /// let v3 = Vector::new(vec![1.0, 2.0, 4.0]);
    ///
    /// assert!(v1.eq(&v2));
    /// assert!(!v1.eq(&v3));
    /// ```
    fn eq(&self, other: &Self) -> bool {
        if self.components.len() != other.components.len() {
            return false;
        }
        for i in 0..self.components.len() {
            if self[i] != other[i] {
                return false;
            }
        }
        true
    }
}


impl<T: Display> Display for Vector<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "[")?;
        for (i, c) in self.components.iter().enumerate() {
            write!(f, "{c}")?;
            if i != self.components.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")
    }
}

impl<T> PartialOrd for Vector<T>
    where
        T: Default + Copy + std::ops::Mul<Output=T> + PartialOrd + std::ops::Sub<Output=T> + std::ops::Add<Output=T> + std::ops::Div<Output=T> + std::ops::Neg<Output=T> + std::fmt::Debug + Float,
{
    /// Compares the magnitude (length) of two vectors and returns an `Option` that
    /// represents their ordering relationship.
    ///
    /// If `self` is greater than `other`, returns `Some(Ordering::Greater)`.
    /// If `self` is less than `other`, returns `Some(Ordering::Less)`.
    /// If `self` and `other` are equal, returns `Some(Ordering::Equal)`.
    /// If either vector contains `NaN` values, returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::cmp::Ordering;
    /// use rusty_linear_algebra::math::Vector;
    /// let a = Vector::new(vec![3.0, 4.0]);
    /// let b = Vector::new(vec![1.0, 2.0]);
    /// let c = Vector::new(vec![6.0, 8.0]);
    ///
    /// assert_eq!(a.partial_cmp(&b), Some(Ordering::Greater));
    /// assert_eq!(a.partial_cmp(&c), Some(Ordering::Less));
    /// assert_eq!(a.partial_cmp(&a), Some(Ordering::Equal));
    /// ```
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.magnitude().partial_cmp(&other.magnitude())
    }
}

impl<T> Index<usize> for Vector<T> {
    type Output = T;
    /// Returns the i-th component of the vector.
    ///
    /// # Arguments
    ///
    /// * `i` - The index of the component to retrieve, where 0 represents the first component.
    ///
    /// # Panics
    ///
    /// Panics if `i` is greater than or equal to the dimension of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusty_linear_algebra::math::Vector;
    /// let v = Vector::new(vec![1.0, 2.0, 3.0]);
    ///
    /// assert_eq!(v[0], 1.0);
    /// assert_eq!(v[1], 2.0);
    /// assert_eq!(v[2], 3.0);
    /// ```
    fn index(&self, index: usize) -> &Self::Output {
        &self.components[index]
    }
}

impl<T> std::ops::Add<Vector<T>> for Vector<T> where T: std::ops::Add<Output=T> + Copy {
    type Output = Vector<T>;

    /// Adds two vectors component-wise, returning a new vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_linear_algebra::math::Vector;
    ///
    /// let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
    /// let v2 = Vector::new(vec![4.0, 5.0, 6.0]);
    /// let result = v1 + v2;
    /// assert_eq!(result, Vector::new(vec![5.0, 7.0, 9.0]));
    /// ```
    ///
    /// # Arguments
    ///
    /// * `self` - The first vector to add.
    /// * `other` - The second vector to add.
    ///
    /// # Returns
    ///
    /// A new `Vector` that is the sum of the two input vectors.
    ///
    /// # Panics
    ///
    /// This method will panic if the two input vectors do not have the same number of components.
    fn add(self, rhs: Vector<T>) -> Self::Output {
        assert_eq!(self.components.len(), rhs.components.len(), "Cannot add vectors of different sizes.");
        let res = self.components.iter().zip(rhs.components.iter()).map(|(&a, &b)| a + b).collect();
        Vector { components: res }
    }
}

impl<T> std::ops::Sub<Vector<T>> for Vector<T> where T: std::ops::Sub<Output=T> + Copy {
    type Output = Vector<T>;
    /// Returns a new Vector<T> that is the result of subtracting another Vector<T> from this one.
    ///
    /// # Arguments
    ///
    /// * `other` - Another Vector<T> to subtract from this Vector<T>.
    ///
    /// # Panics
    ///
    /// This method will panic if the other Vector<T> is not the same size as this one.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusty_linear_algebra::math::Vector;
    /// let a = Vector::new(vec![1.0, 2.0, 3.0]);
    /// let b = Vector::new(vec![0.5, 1.0, 1.5]);
    /// let c = a - b;
    /// assert_eq!(c, Vector::new(vec![0.5, 1.0, 1.5]));
    /// ```
    fn sub(self, rhs: Vector<T>) -> Self::Output {
        assert_eq!(self.components.len(), rhs.components.len(), "Cannot subtract vectors of different sizes.");
        let res = self.components.iter().zip(rhs.components.iter()).map(|(&a, &b)| a - b).collect();
        Vector { components: res }
    }
}

impl<T> std::ops::Mul<Vector<T>> for Vector<T> where T: std::ops::Mul<Output=T> + Copy {
    type Output = Vector<T>;

    /// Multiplies each component of this vector with the corresponding component of the given vector and returns a new vector with the resulting components.
    ///
    /// # Arguments
    ///
    /// * `other` - The vector to multiply this vector with.
    ///
    /// # Example
    ///
    /// ```
    /// # use rusty_linear_algebra::math::Vector;
    /// let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
    /// let v2 = Vector::new(vec![4.0, 5.0, 6.0]);
    /// let result = v1 * v2;
    /// assert_eq!(result, Vector::new(vec![4.0, 10.0, 18.0]));
    /// ```
    fn mul(self, rhs: Vector<T>) -> Self::Output {
        assert_eq!(self.components.len(), rhs.components.len(), "Cannot multiply vectors of different sizes.");
        let res = self.components.iter().zip(rhs.components.iter()).map(|(&a, &b)| a * b).collect();
        Vector { components: res }
    }
}

impl<T> Vector<T>
    where T:
    std::ops::Mul<Output=T> +
    std::ops::Add<Output=T> +
    std::ops::Div<Output=T> +
    Sized +
    Default +
    Copy +
    Float
{
    /// Creates a new vector with the given components.
    ///
    /// # Arguments
    ///
    /// * `components`: A vector of values of the same type `T` that represent the components of the new vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_linear_algebra::math::Vector;
    ///
    /// let v = Vector::new(vec![1.0, 2.0, 3.0]);
    /// assert_eq!(v[0], 1.0);
    /// assert_eq!(v[1], 2.0);
    /// assert_eq!(v[2], 3.0);
    /// ```
    ///
    /// # Remarks
    ///
    /// The number of components in the vector is determined by the length of the input vector. If the input vector is empty, an empty vector will be created.
    pub fn new(comp: Vec<T>) -> Self {
        Self { components: comp }
    }

    /// Returns the magnitude (i.e., the length or the norm) of the vector.
    ///
    /// The magnitude of a vector is the square root of the sum of the squares of its components.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_linear_algebra::math::Vector;
    ///
    /// let v = Vector::new(vec![3.0, 4.0, 0.0]);
    /// assert_eq!(v.magnitude(), 5.0);
    ///
    /// let v = Vector::new(vec![1.0, 1.0, 1.0, 1.0]);
    /// assert_eq!(v.magnitude(), 2.0);
    /// ```
    ///
    /// # Panics
    ///
    /// This method will panic if the type of the vector's components does not implement the `std::ops::Mul` and `std::ops::Add` traits.
    ///
    /// # Remarks
    ///
    /// The magnitude of a vector is a measure of its "length" or "size". It is always a non-negative value.
    pub fn magnitude(&self) -> T {
        self.components
            .iter()
            .map(|c| *c * *c)
            .fold(T::default(), |sum, c| sum + c)
            .sqrt()
    }

    /// Normalizes the vector to have a length of 1.
    ///
    /// This method divides each component of the vector by its magnitude, so that the resulting vector has a length of 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_linear_algebra::math::Vector;
    ///
    /// let mut v = Vector::new(vec![3.0, 4.0]);
    /// let u = v.normalize();
    /// assert_eq!(u.magnitude(), 1.0);
    ///
    /// let mut v = Vector::new(vec![1.0, 1.0, 1.0]);
    /// let u = v.normalize();
    /// assert_eq!(u.magnitude(), 1.0);
    /// ```
    ///
    /// # Panics
    ///
    /// This method will panic if the length of the vector is zero.
    ///
    /// # Remarks
    ///
    /// Normalizing a vector is a way to obtain a unit vector in the same direction as the original vector. It is often used in various applications to remove the effect of scale and only retain the direction information of a vector.
    pub fn normalize(&mut self) {
        let magnitude = self.magnitude();
        for c in self.components.iter_mut() {
            *c = *c / magnitude;
        }
    }

    /// Creates a new `Vector<T>` instance by cloning the values from a slice of type T.
    ///
    /// # Arguments
    ///
    /// * `slice`: A slice of type T containing the values to be cloned.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_linear_algebra::math::Vector;
    ///
    /// let v1 = Vector::from_slice(&[1.0, 2.0, 3.0]);
    ///
    /// assert_eq!(v1, Vector::new(vec![1.0, 2.0, 3.0]));
    /// ```
    ///
    pub fn from_slice(vec: &[T]) -> Vector<T> {
        Vector::new(vec.to_vec())
    }

    /// Creates a new `Vector<T>` instance by cloning the elements of the given `Vec<T>`.
    ///
    /// # Arguments
    ///
    /// * `vec` - A reference to a `Vec<T>` containing the elements to be cloned.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_linear_algebra::math::Vector;
    ///
    /// let vec = vec![1.0, 2.0, 3.0];
    /// let cloned_vec = Vector::<f32>::from_vec(&vec);
    ///
    /// assert_eq!(cloned_vec, Vector::new(vec![1.0, 2.0, 3.0]));
    /// ```
    pub fn from_vec(vec: &Vec<T>) -> Vector<T> {
        Vector::new(vec.clone())
    }
}

#[cfg(test)]
mod tests {
    use crate::math::Vector;

    #[test]
    #[should_panic]
    fn test_sub_vector_of_different_sizes() {
        let v1: Vector<_> = vector!(1.0,2.0,3.0);
        let v2: Vector<_> = vector!(1.0, 2.0);
        let v3 = v2 - v1;
    }

    #[test]
    #[should_panic]
    fn test_add_vector_of_different_sizes() {
        let v1: Vector<_> = vector!(1.0,2.0,3.0);
        let v2: Vector<_> = vector!(1.0, 2.0);
        let _ = v2 + v1;
    }

    #[test]
    fn test_vector_macro() {
        let vec1 = vector![1.0, 2.0, 3.0];
        let vec2 = vector!(1,2,3);

        assert_eq!(vec1.components, vec![1.0, 2.0, 3.0]);
        assert_eq!(vec2.components, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_try_from_string() {
        let v1 = Vector::try_from(String::from("[1.0,2.0,3.0]")).unwrap();
        let v2 = Vector::try_from(String::from("[1.0, 2.0, 3.0]")).unwrap();
        let v3 = Vector::<f64>::try_from(String::from("[1.0,2.0,3.0,4.0]"));
        assert_eq!(v1, Vector::new(vec![1.0, 2.0, 3.0]));
        assert_eq!(v1, v2);
    }
}