"""
vec.py - Vector algebra utilities for mathematical operations on vectors.

Provides a Vector class supporting arithmetic, dot/cross products, normalization, angle calculation, and more.
"""
import math
class Vector:
    """
    A class to represent mathematical vectors of arbitrary dimension.

    Supports vector arithmetic, dot and cross products, normalization, angle calculation, and more.
    Components are stored as a list of numbers.
    """
import math

class Vector:
    def __init__(self, *args):
        self.components = list(args)

    @property
    def x(self):
        """Allows access to the first component (v.x)."""
        if len(self.components) > 0:
            return self.components[0]
        raise IndexError("Vector does not have an x component.")
    
    @x.setter
    def x(self, value):
        """Allows setting the first component (v.x = value)."""
        if len(self.components) > 0:
            self.components[0] = value
        else:
            raise IndexError("Vector does not have an x component to set.")
    
    @property
    def y(self):
        """Allows access to the second component (v.y)."""
        if len(self.components) > 1:
            return self.components[1]
        raise IndexError("Vector does not have a y component.")
    
    @y.setter
    def y(self, value):
        """Allows setting the second component (v.y = value)."""
        if len(self.components) > 1:
            self.components[1] = value
        else:
            raise IndexError("Vector does not have a y component to set.")
    
    @property
    def z(self):
        """Allows access to the third component (v.z)."""
        if len(self.components) > 2:
            return self.components[2]
        raise IndexError("Vector does not have a z component.")
    
    @z.setter
    def z(self, value):
        """Allows setting the third component (v.z = value)."""
        if len(self.components) > 2:
            self.components[2] = value
        else:
            raise IndexError("Vector does not have a z component to set.")

    @property
    def dimension(self):
        """Returns the dimension of the vector."""
        return len(self.components)

    def __len__(self):
        return len(self.components)

    def __getitem__(self, index):
        return self.components[index]

    def __add__(self, other):
        if len(self.components) != len(other.components):
            raise ValueError("Vectors must have the same dimension")
        return Vector(*(a + b for a, b in zip(self.components, other.components)))

    def __sub__(self, other):
        if len(self.components) != len(other.components):
            raise ValueError("Vectors must have the same dimension")
        return Vector(*(a - b for a, b in zip(self.components, other.components)))

    def __mul__(self, scalar):
        if isinstance(scalar, int) or isinstance(scalar, float):
            return Vector(*(a * scalar for a in self.components))
        elif isinstance(scalar, Vector):
            if len(self.components) != len(scalar.components):
                raise ValueError("Vectors must have the same dimension")
            return sum(*(a * b for a, b in zip(self.components, scalar.components)))

    
    def __rmul__(self, scalar):
        if isinstance(scalar, int) or isinstance(scalar, float):
            return Vector(*(a * scalar for a in self.components))
        elif isinstance(scalar, Vector):
            if len(self.components) != len(scalar.components):
                raise ValueError("Vectors must have the same dimension")
            return sum(*(a * b for a, b in zip(self.components, scalar.components)))
        
    def dot(self, other):
        if len(self.components) != len(other.components):
            raise ValueError("Vectors must have the same dimension")
        return sum(a * b for a, b in zip(self.components, other.components))
    
    def __truediv__(self, scalar):
        if scalar == 0:
            raise ZeroDivisionError("Division by zero is not allowed")
        return Vector(*(a / scalar for a in self.components))
    
    def __floordiv__(self, scalar):
        if scalar == 0:
            raise ZeroDivisionError("Division by zero is not allowed")
        return Vector(*(a // scalar for a in self.components))
    
    def __mod__(self, scalar):
        if scalar == 0:
            raise ZeroDivisionError("Division by zero is not allowed")
        return Vector(*(a % scalar for a in self.components))

    def __matmul__(self, other):
        """
        Implements the cross product for 3D vectors using the @ operator.
        Result is a new Vector.
        """
        # The cross product is only defined for 3-dimensional vectors.
        if len(self.components) != 3 or len(other.components) != 3:
            raise ValueError("Cross product is only defined for 3-dimensional vectors")

        x1, y1, z1 = self.components
        x2, y2, z2 = other.components

        # Cross product formula:
        # A x B = (A_y * B_z - A_z * B_y,
        #          A_z * B_x - A_x * B_z,
        #          A_x * B_y - A_y * B_x)
        result_x = (y1 * z2) - (z1 * y2)
        result_y = (z1 * x2) - (x1 * z2)
        result_z = (x1 * y2) - (y1 * x2)

        return Vector(result_x, result_y, result_z)
    
    def __eq__(self, other):
        if isinstance(other, Vector):
        # Use zip to compare components
            return self.components == other.components
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
    
    @property
    def magnitude(self):
        return sum(comp**2 for comp in self.components)**0.5

    def angle_with(self, other):
        """
        Returns the angle in radians between this vector and another vector.
        Raises ValueError if vectors have different dimensions or if either is zero vector.
        """
        if not isinstance(other, Vector):
            raise TypeError("angle_with() argument must be a Vector")
        if len(self.components) != len(other.components):
            raise ValueError("Vectors must have the same dimension")
        mag_self = self.magnitude
        mag_other = other.magnitude
        if mag_self == 0 or mag_other == 0:
            raise ValueError("Cannot compute angle with zero-length vector")
        dot_prod = self.dot(other)
        # Clamp value to [-1, 1] to avoid domain errors due to floating point
        cos_theta = max(min(dot_prod / (mag_self * mag_other), 1.0), -1.0)
        return math.acos(cos_theta)
# filepath: vscode-vfs://github/maincodepy-htmlbasic/mathlibpy/vec.py

    def normalize(self):
        """
        Returns a unit vector (vector with magnitude 1) in the same direction as this vector.
        Raises ValueError if the vector is zero.
        """
        mag = self.magnitude
        if mag == 0:
            raise ValueError("Cannot normalize the zero vector")
        return Vector(*(a / mag for a in self.components))

    def __lt__(self, other):
        if isinstance(other, Vector):
            return self.magnitude < other.magnitude
        raise TypeError(f"'<' not supported between 'Vector' and '{type(other).__name__}'")

    def __le__(self, other):
        if isinstance(other, Vector):
            return self.magnitude <= other.magnitude
        raise TypeError(f"'<=' not supported between 'Vector' and '{type(other).__name__}'")

    def __gt__(self, other):
        if isinstance(other, Vector):
            return self.magnitude > other.magnitude
        raise TypeError(f"'>' not supported between 'Vector' and '{type(other).__name__}'")

    def __ge__(self, other):
        if isinstance(other, Vector):
            return self.magnitude >= other.magnitude
        raise TypeError(f"'>=' not supported between 'Vector' and '{type(other).__name__}'")


    def __neg__(self):
        return Vector(*(-a for a in self.components))

    def __repr__(self):
        return f"Vector({', '.join(map(str, self.components))})"
    
    def __str__(self):
        return f"<{', '.join(map(str, self.components))}>"
    
    @property
    def is_zero(self):
        """Checks if the vector is a zero vector."""
        return all(comp == 0 for comp in self.components)

    def __iter__(self):
        """Allows iteration over the vector's components."""
        return iter(self.components)

    def __hash__(self):
        """Allows Vector to be used as a dictionary key or in sets (if immutable)."""
        return hash(tuple(self.components))

class Vector3D(Vector):
    """
    A class to represent 3-dimensional vectors, inheriting from Vector.
    
    Provides additional properties for x, y, z components and methods specific to 3D vectors.
    """
    def __init__(self, x=0, y=0, z=0):
        super().__init__(x, y, z)

    @property
    def x(self):
        return self.components[0]

    @property
    def y(self):
        return self.components[1]

    @property
    def z(self):
        return self.components[2]
    
    def __repr__(self):
        """Returns a string representation of the vector in 3D."""
        if self.z != 0:
            return f"[{self.x}, {self.y}, {self.z}]"
        if self.y != 0 and self.z == 0:
            return f"[{self.x}, {self.y}]"
        if self.x != 0 and self.y == 0 and self.z == 0:
            return f"[{self.x}]"
        else:
            return f"[0, 0, 0]"
    
    def __str__(self):
        if self.x != 0 and self.y != 0 and self.z != 0:
            return f"{self.x}i + {self.y}j + {self.z}k"
        elif self.x != 0 and self.y != 0 and self.z == 0:
            return f"{self.x}i + {self.y}j"
        elif self.x != 0 and self.y == 0 and self.z != 0:
            return f"{self.x}i + {self.z}k"
        elif self.x == 0 and self.y != 0 and self.z != 0:
            return f"{self.y}j + {self.z}k"
        elif self.x != 0 and self.y == 0 and self.z == 0:
            return f"{self.x}i"
        elif self.x == 0 and self.y != 0 and self.z == 0:
            return f"{self.y}j"
        elif self.x == 0 and self.y == 0 and self.z != 0:
            return f"{self.z}k"
    

def magnitude(self):
    if isinstance(self, Vector):
        return sum(comp**2 for comp in self.components)**0.5
    else:
        raise TypeError(f"magnitude() is only defined for Vector instances, not {type(self).__name__}")

