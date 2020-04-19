#include <iostream>

using namespace std;

struct node
{
	node *left;
	node *right;
	int data;
};

class BT
{
	node *root;
public:
	bt(){
		root=NULL;
	}
	int isEmpty(){
		return (root==NULL);
	}
	void insert(int item);
	void displayBT();
	void printBT(node *)
};