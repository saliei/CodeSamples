#include <iostream>

bool are_equal(const double d1, const double d2, const double epsilon = 0.001)
{
    return std::fabs(d1 - d2) < epsilon;
}

namespace temperature
{
    enum class scale { celsius, fahrenheit, kelvin };
    template<scale S>
    class quantity
    {
        const double amount;
        public:
            constexpr explicit quantity(const double a): amount(a) {}
            explicit operator double() const { return amount; }
    };

    template<scale S>
    inline bool operator==(quantity<S> const &lhs, quantity<S> const &rhs)
    {
        return are_equal(static_cast<double>(lhs), static_cast<double>(rhs));
    }

    template<scale S>
        inline bool operator!=(quantity<S> const &lhs, quantity<S> const &rhs)
        {
            return !(lhs == rhs);
        }

    template<scale S>
        inline bool operator<(quantity<S> const &lhs, quantity<S> const &rhs)
        {
            return static_cast<double>(lhs) < static_cast<double>(rhs);
        }

    template<scale S>
        inline bool operator>(quantity<S> const &lhs, quantity<S> const &rhs)
        {
            return rhs < lhs;
        }

    template<scale S>
        inline bool operator<=(quantity<S> const &lhs, quantity<S> const &rhs)
        {
            return !(lhs > rhs);
        }

    template<scale S>
        inline bool operator>=(quantity<S> const &lhs, quantity<S> const &rhs)
        {
            return !(lhs < rhs);
        }
    
    template<scale S>
        constexpr quantity<S> operator+(quantity<S> const &q1, quantity<S> const &q2)
        {
            return quantity<S>(static_cast<double>(q1) + static_cast<double>(q2));
        }

    template<scale S>
        constexpr quantity<S> operator-(quantity<S> const &q1, quantity<S> const &q2)
        {
            return quantity<S>(static_cast<double>(q1) - static_cast<double>(q2));
        }
}
