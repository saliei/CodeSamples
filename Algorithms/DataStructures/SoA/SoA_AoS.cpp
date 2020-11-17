#include <vector>
#include <iostream>
#include <chrono>
#include <string>
#include <numeric>
#include <iomanip>
#include <ctime>

const uint32_t ELEM=(1<<20);

/***********/
/*  A o S  */
/***********/
struct Person {
    Person(const std::string& n, uint8_t a,  uint32_t d):
        name(n), age(a), dob(d) {}

    std::string name;
    uint8_t age;
    uint32_t dob;
};

void addPerson(std::vector<Person>& v, Person&& p){
    v.push_back(std::move(p));
}

uint64_t averageNameLen(const std::vector<Person>& v){
    return std::accumulate(v.begin(), v.end(), (uint8_t)0, 
            [](uint64_t sum, auto& p){return sum + p.name.length();}) / v.size();
}

uint64_t averageAge(const std::vector<Person>& v){
    return std::accumulate(v.begin(), v.end(), (uint8_t)0, 
            [](uint8_t sum, auto& p){ return sum + p.age; }) / v.size();
}

uint64_t averageDob(const std::vector<Person>& v){
    return std::accumulate(v.begin(), v.end(), (uint8_t)0, 
            [](uint8_t sum, auto& p){ return sum + p.dob; }) / v.size();
}


/***********/
/*  S o A  */
/***********/
struct Persons{
    std::vector<std::string> names;
    std::vector<uint8_t> ages;
    std::vector<uint32_t> dobs;

    void addPerson(std::string n, uint8_t a, uint32_t d){
        names.push_back(n);
        ages.push_back(a);
        dobs.push_back(d);
    }

    uint64_t averageNameLen() const{
        return std::accumulate(std::begin(names), std::end(names), (uint8_t)0, 
                [](auto sum, auto& n){ return sum + n.length(); }) / names.size();
    }

    uint64_t averageAge() const{
        return std::accumulate(std::begin(ages), std::end(ages), (uint8_t)0,
                [](auto sum, auto& a){ return sum + a; }) / ages.size();
    }

    uint64_t averageDob() const{
        return std::accumulate(std::begin(dobs), std::end(dobs), (uint8_t)0, 
                [](auto sum, auto& d){ return sum + d; }) / dobs.size();
    }
};

int main(int argc, char** argv){

    std::vector<Person> aos_p;
    aos_p.reserve(ELEM);
    for(int i = 0; i < ELEM; i++){
        addPerson(aos_p, Person(std::string("RAND NAME"), i % 0xFF, i % 0xFFFF));
    }
    auto start_t = std::chrono::high_resolution_clock::now();
    auto sum = averageNameLen(aos_p);
    sum += averageAge(aos_p);
    sum += averageDob(aos_p);
    auto end_t = std::chrono::high_resolution_clock::now();
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << sum ;
    std::cout <<  "AoS Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start_t).count() / 1000.f << "ms \n";
    aos_p.clear();
    aos_p.shrink_to_fit();

    Persons soa_p;
    soa_p.names.reserve(ELEM);
    soa_p.ages.reserve(ELEM);
    soa_p.dobs.reserve(ELEM);

    std::srand(std::time(nullptr));
    for(int i = 0; i < ELEM; i++){
        soa_p.addPerson(std::string("RAND NAME"), std::rand() % 0xFF, std::rand() % 0xFFFF);
    }

    start_t = std::chrono::high_resolution_clock::now();
    sum = soa_p.averageNameLen();
    sum += soa_p.averageAge();
    sum += soa_p.averageDob();
    end_t = std::chrono::high_resolution_clock::now();
    
    //std::cout << std::fixed << std::setprecision(3);
    std::cout << "SoA Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start_t).count() / 1000.f << "ms \n";

    return sum;

    

}
