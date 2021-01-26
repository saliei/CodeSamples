public class Employee
{
    String name;
    int age;
    String designation;
    double salary;

    public Employee(String name)
    {
        this.name = name;
    }

    public void setAge(int empAge)
    {
        age = empAge;
    }

    public void setDgn(String empDgn)
    {
        designation = empDgn;
    }

    public void setSal(double empSal)
    {
        salary = empSal;
    }

    public void printEmployee()
    {
        System.out.println("Name: " + name);
        System.out.println("Age: " + age);
        System.out.println("Designation: " + designation);
        System.out.println("Salary: " + salary);
    }
}
