#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char **argv)
{
    std::ofstream myfile;
    myfile.open("example.txt", std::ios::out | std::ios::app);
    if(myfile.is_open())
    {
        myfile << "Writing this with a stream to \n a file!\n";
        myfile.close();
    }
    else std::cout << "Unable to open file!" << std::endl;

    std::string line;
    std::ifstream ifile("sample.txt");
    //ifile.open("sample.txt", std::ios::in);
    if(ifile.is_open())
    {
        while(std::getline(ifile, line))
            std::cout << line << std::endl;
        ifile.close();
    }
    else std::cout << "Unable to open file!" << std::endl;


    return 0;
}
