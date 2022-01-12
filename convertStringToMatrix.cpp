#include <iostream>
#include <fstream>
#include <string>


#include <cstring>  
using namespace std;
int main(){
   
   string cam0_values;
   string cam1_values;
   bool enter = false
   
   // reading the cam0 values
   for(int i=0;i<cam0.length();i++){ //taking values inside the square brackets to a string
   if(enter){
   	if(cam0[i]=="]"){
    	enter=false;
    }
    else{
    	cam0_values += cam0[i];
    }
   }
   if(cam0[i]=='[')	{
   		enter = true;
   	}
   }
   
	 std::string segment;
	 std::vector<std::string> cam0list;
   int cam0data[3][3];

	 while(std::getline(cam0_values, segment, ';')){ // spliting the string with semi-colon
   	cam0list.push_back(segment); //Spit string at ';' character
	 }
   
  for (size_t i = 0; i < cam0list.size(); i++) { // filling the matrix values
        for(int j = 0; j < cam0list[i].length(); i+=2){
        	cam0data[i][j/2] = (int)cam0list[i].at(j);
        }
  }
   
  
   
   
   
   // reading the cam1 values
   for(int i=0;i<cam1.length();i++){ //taking values inside the square brackets to a string
   if(enter){
   	if(cam1[i]=="]"){
    	enter=false;
    }
    else{
    	cam1_values += cam1[i];
    }
   }
   if(cam1[i]=='[')	{
   		enter = true;
   	}
   }
   
   std::string segment;
	 std::vector<std::string> cam1list;

	 while(std::getline(cam1_values, segment, ';')){ // spliting the string with semi-colon
   	cam1list.push_back(segment); //Spit string at ';' character
	 }
   int cam1data[3][3];
   for (size_t i = 0; i < cam1list.size(); i++) { // filling the matrix values
        for(int j = 0; j < cam1list[i].length(); i+=2){
        	cam1data[i][j/2] = (int)cam1list[i].at(j);
        }
  }
   
   cv::Mat cam0Mat = cv::Mat(3, 3, CV_32F, cam0data); // matrix to be used 
   cv::Mat cam1Mat = cv::Mat(3, 3, CV_32F, cam1data); // matrix to be used 
   
   
   
   
   
}