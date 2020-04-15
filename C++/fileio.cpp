#include <iostream>
#include <string>
#include <fstream>

using namespace std;

int main(int argc, char** argv)
{
    ofstream writeToFile;
    ifstream readFrmFile;
    string writeStr = "";
    string readStr = "";

    /*
     *writeToFile.open("test.txt", ios_base::out | ios_base::trunc);
     *if(writeToFile.is_open())
     *{
     *    writeToFile << "Begining of file.\n";
     *    cout << "Enter String to write to file: ";
     *    getline(cin, writeStr);
     *    writeToFile << writeStr;
     *    writeToFile.close();
     *}
     */

    readFrmFile.open("test.txt", ios_base::in);
    if(readFrmFile.is_open())
    {
        while(readFrmFile.good())
        {
            getline(readFrmFile, readStr);
            cout << readStr << endl;
        }
    }
    readFrmFile.close();


}
