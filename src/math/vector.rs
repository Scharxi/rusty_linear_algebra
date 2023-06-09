use std::cmp::Ordering;
use std::fmt::{Display, Formatter};
use std::ops::{Index, Mul};
use num_traits::{Float};

type Point3D<T> = (T, T, T);
type Point2D<T> = (T, T);

macro_rules! vector {
    ($($x:expr),*) => {
        Vector::new(vec![$($x as f64),*])
    };
}

#[derive(Clone, Debug)]
pub struct Vector<T> {
    components: Vec<T>,
}

impl<T> IntoIterator for Vector<T> {
    type Item = T;
    type IntoIter = VectorIterator<T>;

    /// Returns an iterator that takes ownership of the vector and returns the
    /// elements in sequence. This iterator will consume the vector, which means
    /// that once the iterator has been created, the original vector can no longer
    /// be accessed unless the iterator is consumed.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_linear_algebra::math::Vector;
    ///
    /// let v = Vector::new(vec![1.0, 2.0, 3.0]);
    /// let mut iter = v.into_iter();
    ///
    /// assert_eq!(iter.next(), Some(1.0));
    /// assert_eq!(iter.next(), Some(2.0));
    /// assert_eq!(iter.next(), Some(3.0));
    /// assert_eq!(iter.next(), None);
    /// ```
    fn into_iter(self) -> Self::IntoIter {
        VectorIterator { vector: self }
    }
}

pub struct VectorIterator<T> {
    vector: Vector<T>,
}

impl<T> Iterator for VectorIterator<T> {
    type Item = T;

    /// Returns the next item in the vector iterator or `None` if the iterator is empty.
    fn next(&mut self) -> Option<Self::Item> {
        if self.vector.components.is_empty() {
            None
        } else {
            Some(self.vector.components.remove(0))
        }
    }
}

impl<T: std::str::FromStr> TryFrom<&str> for Vector<T> {
    type Error = &'static str;

    /// Attempts to create a new `Vector<T>` from a string slice.
    ///
    /// The string slice should contain a comma-separated list of values enclosed in square brackets,
    /// representing the components of the vector. For example, the string "[1.0, 2.0, 3.0]" would create
    /// a new `Vector<f64>` with three components.
    ///
    /// # Errors
    ///
    /// Returns a `ParseError` if the string slice is not in the correct format or if any of the values
    /// cannot be parsed as type `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::convert::TryFrom;
    /// use rusty_linear_algebra::math::Vector;
    ///
    /// let vector = Vector::<f64>::try_from("[1.0, 2.0, 3.0]").unwrap();
    /// assert_eq!(vector.components(), &[1.0, 2.0, 3.0]);
    /// ```
    ///
    /// This method is typically used to parse user input or configuration files where vector
    /// components are specified as strings.
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let components = value
            .trim_matches(|p| p == '[' || p == ']')
            .split(',')
            .map(|s| s.trim().parse::<T>())
            .collect::<Result<Vec<T>, _>>()
            .map_err(|_| "Could not create Vector from string slice.")?;

        Ok(Vector { components })
    }
}

impl<T: std::str::FromStr> TryFrom<String> for Vector<T> {
    type Error = &'static str;

    /// Attempts to create a new `Vector<T>` instance from a string slice. The string should
    /// contain a comma-separated list of numeric values enclosed in square brackets, such as
    /// "[1.0, 2.0, 3.0]". This function returns an error if any of the values in the string
    /// are not numeric or if the string does not have the correct format.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::convert::TryFrom;
    /// use rusty_linear_algebra::math::Vector;
    ///
    /// let v = Vector::<f64>::try_from("[1.0, 2.0, 3.0]".to_string()).unwrap();
    /// assert_eq!(v.len(), 3);
    /// assert_eq!(v[0], 1.0);
    ///
    /// let result = Vector::<f64>::try_from("[1.0, 2.0, foo]".to_string());
    /// assert!(result.is_err());
    /// ```
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
        T: Default + Copy + Mul<Output=T> + PartialOrd + std::ops::Sub<Output=T> + std::ops::Add<Output=T> + std::ops::Div<Output=T> + std::ops::Neg<Output=T> + std::fmt::Debug + Float,
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

impl<T> Mul<Vector<T>> for Vector<T> where T: Mul<Output=T> + Copy {
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

impl<T: std::ops::Mul<Output=T> + Clone> std::ops::Mul<T> for Vector<T> {
    type Output = Self;
    /// Multiplies each component of the vector by a scalar value and returns a new vector.
    ///
    /// # Arguments
    ///
    /// * `scalar`: A scalar value of type T, which will be used to multiply each component of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_linear_algebra::math::Vector;
    ///
    /// let vector = Vector::new(vec![1.0, 2.0, 3.0]);
    /// let scaled_vector = vector * 2.0;
    /// assert_eq!(scaled_vector, Vector::new(vec![2.0, 4.0, 6.0]));
    /// ```
    ///
    /// # Panics
    ///
    /// If any of the vector components cannot be multiplied by the scalar value, this method will panic.
    fn mul(self, other: T) -> Self::Output {
        let mut new_components = Vec::with_capacity(self.components.len());
        for comp in self.components.iter() {
            new_components.push(comp.clone() * other.clone());
        }
        Self { components: new_components }
    }
}

impl<T> Vector<T>
    where T:
    Mul<Output=T> +
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
    /// v.normalize();
    /// assert_eq!(v.magnitude(), 1.0);
    ///
    /// let mut v = Vector::new(vec![1.0, 1.0, 1.0]);
    /// v.normalize();
    /// assert_eq!(v.magnitude(), 1.0);
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

    /// Multiplies each component of the vector by a scalar value and returns a new vector.
    ///
    /// # Arguments
    ///
    /// * `scalar`: A scalar value of type T, which will be used to multiply each component of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_linear_algebra::math::Vector;
    ///
    /// let vector = Vector::new(vec![1.0, 2.0, 3.0]);
    /// let scaled_vector = vector.scalar_mul(2.0);
    /// assert_eq!(scaled_vector, Vector::new(vec![2.0, 4.0, 6.0]));
    /// ```
    ///
    /// # Panics
    ///
    /// If any of the vector components cannot be multiplied by the scalar value, this method will panic.
    pub fn scalar_mul(&self, scalar: T) -> Vector<T> {
        Self::mul(self.clone(), scalar)
    }

    /// Returns a reference to the vector's underlying component data.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusty_linear_algebra::math::Vector;
    /// let v = Vector::new(vec![1.0, 2.0, 3.0]);
    /// let components = v.components();
    ///
    /// assert_eq!(components, &vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn components(&self) -> &Vec<T> {
        &self.components
    }

    /// Returns the number of elements in the vector.
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// Returns true if the vectors components is empty, false otherwise.
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Creates a Vector<T> from two points in n-dimensional space.
    ///
    /// # Arguments
    ///
    /// * `start` - A point in n-dimensional space as an array of type T representing the starting coordinates.
    /// * `end` - A point in n-dimensional space as an array of type T representing the ending coordinates.
    ///
    /// # Panics
    ///
    /// This method will panic if the two input arrays do not have the same length.
    ///
    /// # Example
    ///
    /// ```
    /// use rusty_linear_algebra::math::Vector;
    ///
    /// let start = (0.0, 0.0, 0.0);
    /// let end = (2.0, 4.0, 5.0);
    /// let vec = Vector::from_points(start, end);
    ///
    /// assert_eq!(vec.components(), &[2.0, 4.0, 5.0]);
    /// ```
    pub fn from_points(start: Point3D<T>, end: Point3D<T>) -> Vector<T> {
        Vector {
            components: vec![end.0 - start.0, end.1 - start.1, end.2 - start.2],
        }
    }

    /// Sets all components of the vector to the given value.
    ///
    /// # Arguments
    ///
    /// * `value`: A value of type `T` to set all components to.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusty_linear_algebra::math::Vector;
    /// let mut v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// v.set(0.0);
    /// assert_eq!(v, Vector::from_slice(&[0.0, 0.0, 0.0]));
    /// ```
    pub fn set(&mut self, value: T) {
        for component in &mut self.components {
            *component = value
        }
    }

    /// Creates a new 3d Vector<T> with all components initialized to zero.
    /// # Examples
    ///
    /// ```
    /// # use rusty_linear_algebra::math::Vector;
    /// let mut zero_vec = Vector::<f64>::zero_3d();
    /// assert_eq!(zero_vec[0], 0.0)
    /// ```
    pub fn zero_3d() -> Vector<f64> {
        vector!(0,0,0)
    }

    /// Checks if the vector is a zero vector, i.e. if all of its components are zero.
    ///
    /// # Example
    ///
    /// ```
    /// use rusty_linear_algebra::math::Vector;
    ///
    /// let zero_vector = Vector::from_slice(&[0.0, 0.0, 0.0]);
    /// let non_zero_vector = Vector::from_slice(&[1.0, 1.0, 1.0]);
    ///
    /// assert!(zero_vector.is_zero());
    /// assert!(!non_zero_vector.is_zero());
    /// ```
    pub fn is_zero(&self) -> bool {
        self.components.iter().all(|&x| x == T::zero())
    }

    /// Checks if the vector is a unit vector, i.e., a vector with a magnitude of 1.
    ///
    /// # Returns
    ///
    /// * `true` if the magnitude of the vector is equal to 1.
    /// * `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_linear_algebra::math::Vector;
    ///
    /// let v1 = Vector::from_vec(&vec![0.6, 0.8]);
    /// assert!(v1.is_unity_vector());
    ///
    /// let v2 = Vector::from_vec(&vec![1.0, 2.0]);
    /// assert!(!v2.is_unity_vector());
    /// ```
    pub fn is_unity_vector(&self) -> bool {
        self.magnitude().eq(&T::one())
    }

    /// Creates a new 3d Vector<T> with all components initialized to one.
    /// # Examples
    ///
    /// ```
    /// # use rusty_linear_algebra::math::Vector;
    /// let mut zero_vec = Vector::<f64>::one_3d();
    /// assert_eq!(zero_vec[0], 1.0)
    /// ```
    pub fn one_3d() -> Vector<f64> {
        vector!(1,1,1)
    }

    /// Computes the cross product of the vector with another vector, returning a new `Vector<T>`
    /// that is perpendicular to both input vectors. The input vector and the other vector must be
    /// of the same dimensionality, and the dimensionality must be either 2 or 3.
    ///
    /// # Arguments
    ///
    /// * `other` - A reference to another `Vector<T>` with the same dimensionality as this one
    ///
    /// # Panics
    ///
    /// This method panics if the input vector and the other vector have different dimensionality,
    /// or if the dimensionality is not 2 or 3.
    pub fn cross_product(&self, other: &Self) -> Option<Self> {
        if self.components.len() != 3 || other.components.len() != 3 {
            return None;
        }

        let ax = self.components[0];
        let ay = self.components[1];
        let az = self.components[2];
        let bx = other.components[0];
        let by = other.components[1];
        let bz = other.components[2];

        Some(Self::new(vec![
            ay * bz - az * by,
            az * bx - ax * bz,
            ax * by - ay * bx,
        ]))
    }

    /// Returns a new Vector<T> that is the unit vector of the current vector,
    /// if the current vector is not a zero vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_linear_algebra::math::Vector;
    ///
    /// let v1 = Vector::from_vec(&vec![1.0, 2.0, 3.0]);
    /// let u = v1.unit_vector();
    /// assert!(u.is_unity_vector())
    /// ```
    pub fn unit_vector(&self) -> Vector<T> {
        let magnitude = self.magnitude();
        if magnitude.is_zero() {
            return self.clone();
        }
        let scale = T::one() / magnitude;
        self.scalar_mul(scale)
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

    #[test]
    fn test_try_from_str_slice() {
        let result = Vector::<f64>::try_from("[1.0, 2.0, 3.0]");
        assert_eq!(result, Ok(Vector::new(vec![1.0, 2.0, 3.0])));

        let result = Vector::<f64>::try_from("[1.0, 2.0, 3.0a]");
        assert!(result.is_err());
    }

    #[test]
    fn test_iterator() {
        let v = vector![1.0, 2.0, 3.0];
        let mut iter = v.into_iter();

        assert_eq!(iter.next(), Some(1.0));
        assert_eq!(iter.next(), Some(2.0));
        assert_eq!(iter.next(), Some(3.0));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_cross_product() {
        let v1 = vector!(1.0, 2.0, 3.0);
        let v2 = vector!(4.0, 5.0, 6.0);
        let result = v1.cross_product(&v2);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), vector!(-3.0, 6.0, -3.0));
    }

    #[test]
    fn test_unity_vector() {
        let v1 = Vector::from_slice(&[1.0, 0.0, 0.0]);
        assert!(v1.is_unity_vector());

        let v2 = Vector::from_slice(&[0.0, 1.0, 0.0]);
        assert!(v2.is_unity_vector());

        let v3 = Vector::from_slice(&[0.0, 0.0, 1.0]);
        assert!(v3.is_unity_vector());

        let v4 = Vector::from_slice(&[1.0, 1.0, 1.0]);
        assert!(!v4.is_unity_vector());

        let v5 = Vector::from_slice(&[0.0, 0.0, 0.0]);
        assert!(!v5.is_unity_vector());
    }
}