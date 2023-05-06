use std::array::IntoIter;
use std::iter::Flatten;
use std::slice::Iter;

pub type Matrix3<T> = Matrix<T, 3, 3>;
pub type Matrix4<T> = Matrix<T, 4, 4>;

#[derive(Debug, Clone)]
/// A generic matrix struct with a fixed number of rows and columns.
///
/// The `Matrix` struct is generic over its element type `T` and its size, which is specified by
/// the type-level constants `ROWS` and `COL`. It provides methods for creating and manipulating
/// matrices, as well as implementing the `Index`, `IndexMut`, and `IntoIterator` traits to allow
/// for indexing and iteration over the matrix's elements.
///
/// # Examples
///
/// Create a new 3x3 matrix with all elements set to 0.0:
///
/// ```
/// # use rusty_linear_algebra::math::Matrix;
/// let matrix = Matrix::<f32, 3,3>::new(0.0);
/// ```
///
/// Access an element in the matrix using indexing:
///
/// ```
/// # use rusty_linear_algebra::math::Matrix;
/// let matrix = Matrix::<f32, 3,3>::new(0.0);
/// let elem = matrix[1][2];
/// ```
pub struct Matrix<T, const ROWS: usize, const COL: usize> {
    pub rows: [[T; COL]; ROWS],
}

impl<T: Default + Copy, const ROWS: usize, const COL: usize> Matrix<T, ROWS, COL> {
    /// Creates a new matrix with the specified default value for each element.
    ///
    /// # Arguments
    ///
    /// * `default_value`: The default value to initialize each element of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusty_linear_algebra::math::Matrix;
    /// let m = Matrix::<f64, 3, 2>::new(0.0);
    /// assert_eq!(m.rows, [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]);
    /// ```
    ///
    /// # Panics
    ///
    /// This method does not panic.
    ///
    /// # Remarks
    ///
    /// The resulting matrix has `ROWS` rows and `COL` columns, and every element is initialized with
    /// the specified default value. The `T` type must implement the `Default` and `Clone` traits.
    ///
    /// If `default_value` is not provided, the `Default::default()` method is called to obtain the
    /// default value for the `T` type.
    ///
    /// # Performance
    ///
    /// This method has a time complexity of O(ROWS * COL), as every element in the matrix is
    /// initialized with the default value.
    pub fn new(initial: T) -> Self {
        Self {
            rows: [[initial; COL]; ROWS],
        }
    }
}

pub type Row<T, const COL: usize> = [T; COL];

impl<'a, T: 'a, const ROWS: usize, const COL: usize> IntoIterator for &'a Matrix<T, ROWS, COL> where T: Copy {
    type Item = (usize, T);
    type IntoIter = Box<dyn Iterator<Item=(usize, T)> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new((0..ROWS).flat_map(move |row| {
            self.rows[row].iter().enumerate().map(move |(_, &element)| (row, element))
        }))
    }
}

impl<'a, T: 'a, const ROWS: usize, const COL: usize> Matrix<T, ROWS, COL> {
    pub fn row_iter(&'a self) -> IntoIter<Iter<'a, T>, ROWS> {
        let row_iters: [Iter<T>; ROWS] = array_init::array_init(|i| self.rows[i].iter());
        row_iters.into_iter()
    }

    pub fn element_iter(&'a self) -> Flatten<IntoIter<Iter<'a, T>, { ROWS }>> {
        self.row_iter().flatten()
    }
}

impl<T, const ROWS: usize, const COL: usize> std::ops::Index<usize> for Matrix<T, ROWS, COL> {
    type Output = [T; COL];

    /// Returns a reference to the array containing the elements of the specified row.
    ///
    /// # Arguments
    ///
    /// * `row`: The index of the row to access.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusty_linear_algebra::math::{Matrix, Matrix3};
    /// let mut matrix: Matrix3<f32> = Matrix::new(0.0);
    ///
    /// // Access the first row of the matrix
    /// let row = matrix[0];
    /// assert_eq!(matrix[0][0], 0.0)
    /// ```
    ///
    /// # Panics
    ///
    /// This method will panic if the specified row index is out of bounds.
    fn index(&self, row: usize) -> &[T; COL] {
        &self.rows[row]
    }
}

impl<T, const ROWS: usize, const COL: usize> std::ops::IndexMut<usize> for Matrix<T, ROWS, COL> {
    /// Returns a mutable reference to the array containing the elements of the specified row.
    ///
    /// # Arguments
    ///
    /// * `row`: The index of the row to access.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusty_linear_algebra::math::{Matrix, Matrix3};
    /// let mut matrix: Matrix3<f32> = Matrix::new(0.0);
    ///
    /// // Set the first element of the second row to 1.0
    /// matrix[1][0] = 1.0;
    /// assert_eq!(matrix[1][0], 1.0);
    ///
    /// // Increment all elements of the third row by 2.0
    /// for elem in &mut matrix[2] {
    ///     *elem += 2.0;
    /// }
    /// assert_eq!(matrix[2], [2.0, 2.0, 2.0])
    /// ```
    ///
    /// # Panics
    ///
    /// This method will panic if the specified row index is out of bounds.
    fn index_mut(&mut self, row: usize) -> &mut [T; COL] {
        &mut self.rows[row]
    }
}

#[cfg(test)]
mod tests {
    use crate::math::{Matrix, Matrix3};

    #[test]
    fn test_matrix_iterator() {
        let matrix: Matrix3<f32> = Matrix::new(0.0);
        for (row, element) in &matrix {
            println!("Element at row {} is: {}", row, element);
        }
    }
}