#include <eigen3/Eigen/Dense>


namespace quat3
{


template<typename T>
class Quaternion3
{

public:

  Quaternion3()
  {
    arr = buf;
  }

  ~Quaternion3() {}

  Quaternion3(T* ptr) :
    buf {T(1), T(0), T(0), T(0)}
  {
    arr = ptr;
  }

  Quaternion3(const Quaternion3<T>& other)
  {
    arr = other.arr;
  }

  Quaternion3(const T& w, const T& x, const T& y, const T& z) :
    buf {w, x, y, z}
  {
    arr = buf;
  }

  template<typename T2>
  Quaternion3<T2> operator*(const Quaternion3<T2> &q2) const
  {
    T2 qw = w()*q2.w() - x()*q2.x() - y()*q2.y() - z()*q2.z();
    T2 qx = w()*q2.x() + x()*q2.w() + y()*q2.z() - z()*q2.y();
    T2 qy = w()*q2.y() - x()*q2.z() + y()*q2.w() + z()*q2.x();
    T2 qz = w()*q2.z() + x()*q2.y() - y()*q2.x() + z()*q2.w();
    return Quaternion3<T2>(qw, qx, qy, qz);
  }

  // overload addition operator as boxplus for a quaternion and a 3-vector
  template<typename T2>
  Quaternion3<T2> operator+(const Eigen::Matrix<T2,3,1>& delta) const
  {
    return *this * Quaternion3<T2>::exp(delta);
  }

  Quaternion3& operator=(const Quaternion3<T>& other){
    if (this != &other)
      memcpy(arr, other.arr, 4 * sizeof(T));
    return *this;
  }

  friend std::ostream& operator<<(std::ostream &os, const Quaternion3<T> &q)
  {
    os << q.w() << "\n" << q.x() << "\n" << q.y() << "\n" << q.z();
    return os;
  }

  static Quaternion3<T> exp(const Eigen::Matrix<T,3,1>& delta)
  {
    const T delta_norm = delta.norm();

    Quaternion3<T> q;
    if (delta_norm < T(1e-6)) // avoid numerical error with approximation
      {
        q.setW(T(1.0));
        q.setX(delta(0) / T(2.0));
        q.setY(delta(1) / T(2.0));
        q.setZ(delta(2) / T(2.0));
      }
    else
      {
        const T delta_norm_2 = delta_norm / T(2.0);
        const T sn = sin(delta_norm_2) / delta_norm;
        q.setW(cos(delta_norm_2));
        q.setX(sn * delta(0));
        q.setY(sn * delta(1));
        q.setZ(sn * delta(2));
      }

    return q;
  }

  T w() const { return *arr; }
  T x() const { return *(arr+1); }
  T y() const { return *(arr+2); }
  T z() const { return *(arr+3); }
  void setW(const T& w) { *arr = w; }
  void setX(const T& x) { *(arr+1) = x; }
  void setY(const T& y) { *(arr+2) = y; }
  void setZ(const T& z) { *(arr+3) = z; }
  T* data() { return arr; }

private:

  T* arr = nullptr;
  T buf[4];

};


} // namespace quat3
