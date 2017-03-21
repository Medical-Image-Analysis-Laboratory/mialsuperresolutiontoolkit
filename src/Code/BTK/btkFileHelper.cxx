#include "btkFileHelper.h"


namespace btk
{

//------------------------------------------------------------------------------------------------

bool FileHelper::FileExist(const std::string &file)
{
    std::fstream fs;

    fs.open(file.c_str());

    return fs.is_open();
}

//------------------------------------------------------------------------------------------------

bool FileHelper::FileExistPrintIt(const std::string &file)
{
    std::fstream fs;

    fs.open(file.c_str());

    if(fs.is_open() == true)
    {
        std::cout<<"File "<<file<<" exist."<<std::endl;
        return true;
    }
    else
    {
        std::cout<<"File "<<file<<" does not exist."<<std::endl;
        return false;
    }
}

//------------------------------------------------------------------------------------------------

void FileHelper::FilesExist(std::vector< std::string > &files, std::vector<bool> &result)
{
    result.resize(files.size());
    std::fstream fs;

    for(unsigned int i = 0; i< files.size(); i++)
    {
        fs.open(files[i].c_str());

        if(fs.is_open() == true)
        {
            result[i] = true;
        }
        else
        {
            result[i] = false;
        }

    }
}

//------------------------------------------------------------------------------------------------

void FileHelper::FilesExistPrintIt(std::vector< std::string > &files, std::vector<bool> &result)
{
    result.resize(files.size());
    std::fstream fs;

    for(unsigned int i = 0; i< files.size(); i++)
    {
        fs.open(files[i].c_str());

        if(fs.is_open() == true)
        {
            result[i] = true;
            std::cout<<"File "<<files[i]<<" exist."<<std::endl;
        }
        else
        {
            result[i] = false;
            std::cout<<"File "<<files[i]<<" does not exist."<<std::endl;
        }

    }
}

//------------------------------------------------------------------------------------------------

unsigned int FileHelper::GetExtensionPosition(std::string filename)
{
    unsigned int      i = 0;
    unsigned int    pos = 0;
    unsigned int length = filename.length();
    char          state = 0;
    bool           pass;

    while(i < length)
    {
        switch(state)
        {
            case 0: // start
                if(filename[i] == '.')
                    state = 1;
                else
                    pos++;
                break;

            case 1: // test extensions
                pass = false;

                if(i+6 == length)
                {
                    std::string tmpString = filename.substr(i,i+5);
                    if(tmpString == "nii.gz")
                        pass = true;
                }
                else if(i+4 == length)
                {
                    std::string tmpString = filename.substr(i,i+3);
                    if(tmpString == "nhdr" || tmpString == "nrrd")
                        pass = true;
                }
                else if(i+3 == length)
                {
                    std::string tmpString = filename.substr(i,i+2);
                    if(tmpString == "nii")
                        pass = true;
                }

                if(pass)
                {
                    i = length-1;
                }
                else
                {
                    state = 0;
                    pos  += 2;
                }
                break;

            default: // error
                state = 0;
                break;
        } // switch

        i++;
    } // while

    return pos;
}

//------------------------------------------------------------------------------------------------

std::string FileHelper::GetRadixOf(std::string filename)
{
    std::string radix = "";

    unsigned int pos = GetExtensionPosition(filename);
    radix            = filename.substr(0,pos);

    return radix;
}

//------------------------------------------------------------------------------------------------

std::string FileHelper::GetExtensionOf(std::string filename)
{
    std::string extension = "";

    unsigned int pos = GetExtensionPosition(filename);
    extension        = filename.substr(pos,filename.length()-1);

    return extension;
}

}
