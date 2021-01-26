import java.io.*;

public class EmployeeTest
{
    public static void main(String args[])
    {
        Employee emp1 = new Employee("John Doe");
        Employee emp2 = new Employee("Jim Smith");

        emp1.setAge(20);
        emp1.setDgn("Secretary");
        emp1.setSal(1000);

        emp2.setAge(30);
        emp2.setDgn("Managment");
        emp2.setSal(3000);

        emp1.printEmployee();
        emp2.printEmployee();
    }
}
