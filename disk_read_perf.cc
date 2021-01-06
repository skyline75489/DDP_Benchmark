#include <ctime>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <chrono>

typedef std::vector<std::string> stringvec;
uint64_t timeSinceEpochMillisec() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

#if _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
void read_directory(const std::string& name, stringvec& v)
{
    std::string pattern(name);
    pattern.append("\\*");
    WIN32_FIND_DATA data;
    HANDLE hFind;
    if ((hFind = FindFirstFile(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE) {
        do {
            v.push_back(data.cFileName);
        } while (FindNextFile(hFind, &data) != 0);
        FindClose(hFind);
    }
}
#else
#include <sys/types.h>
#include <dirent.h>
void read_directory(const std::string& name, stringvec& v)
{
    DIR* dirp = opendir(name.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
        v.push_back(dp->d_name);
    }
    closedir(dirp);
}

#endif
using namespace std;

int main(int argc, char* argv[]) {
    uint64_t start = timeSinceEpochMillisec();

    std::string path = std::string(argv[1]);
    stringvec dirs;
    read_directory(path, dirs);
    uint64_t total_size = 0;
    std::vector<char> buffer;
    buffer.reserve(1024 * 1024 * 30);
    for (stringvec::iterator it = dirs.begin(); it != dirs.end(); it++)
    {
        std::string currDir = *it;
        if (currDir.compare(".") == 0 || currDir.compare("..") == 0)
        {
            continue;
        }
        stringvec subDirs;
        std::string currDir2 = path + "/" + (*it);
        //std::cout << currDir2 << std::endl;
        read_directory(currDir2, subDirs);

        uint64_t i_start = timeSinceEpochMillisec();
        uint64_t size = 0;
        for (stringvec::iterator it = subDirs.begin(); it != subDirs.end(); it++)
        {
            std::string filename = *it;
            if (filename.compare(".") == 0 || filename.compare("..") == 0)
            {
                continue;
            }
            stringvec files;
            std::string currFile = currDir2 + "/" + filename;
            //std::cout << currFile << std::endl;
            std::ifstream f(currFile.c_str(), std::ios::binary );
            f.unsetf(std::ios::skipws);
            std::streampos fileSize;

            f.seekg(0, std::ios::end);
            fileSize = f.tellg();
            f.seekg(0, std::ios::beg);

            if (f.read(buffer.data(), fileSize))
            {
            }
            f.close();
            size_t i_size = fileSize;
            size += i_size;
        }
        uint64_t i_end = timeSinceEpochMillisec();
        uint64_t i_time = i_end - i_start;
        float i_speed = size / (float)i_time * 1000 / 1024 / 1024;
        std::cout << "Folder: " << currDir2 << "\tsize:" << size << "\ttime:" << i_time << "ms" << "\tspeed:" << i_speed << " MBps"<< std::endl;
        total_size += size;
    }

    uint64_t end = timeSinceEpochMillisec();
    uint64_t time = end - start;
    float speed = total_size / (float)time * 1000 / 1024 / 1024;
    std::cout << "Total\tsize:" <<  total_size << "\ttime:" << time <<  "ms\tspeed:" << speed << " MBps" << std::endl;
    return 0;
}
