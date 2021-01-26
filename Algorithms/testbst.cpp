// unbalanced_binary_tree.h
#ifndef UNBALANCED_BINARY_TREE_H
#define UNBALANCED_BINARY_TREE_H

// uncomment this line to enable move semantics; comment it to disable it.
//#define MOVE_SEMANTICS

#include <memory>
#include <initializer_list>
#include <vector>
#include <iostream>
#include <utility>

using namespace std;

namespace unbalanced_binary_tree {
  template <typename T>
  class tree
  {
  public:
  // default constructor. Empty tree.
  tree() : root_{ nullptr }
  { cout << "Creating empty tree (default constructor)." << endl; }

  // this constructor makes a tree from an array of elements.
  tree(const initializer_list<T> elements) : tree{}
  {
    cout << "Populating tree from initializer list (" << elements.size() << " elements)..." << endl;

    for (auto element : elements)
    *this << element;
    }

    // copy constructor.
    tree(const tree<T>& other) : tree{}
    {
      cout << "Beginning tree copy-construction..." << endl;

      // if the other tree isn't empty, it's copied from its root.
      if (!other.is_empty()) root_.reset(new node{*(other.root_)});
    }

#ifdef MOVE_SEMANTICS
    // move constructor
    tree(tree<T>&& other) : root_{ move(other.root_) }
    { cout << "Last tree constructed with move-constructor." << endl; }
#endif // MOVE_SEMANTICS

    // destructor.
    ~tree() { cout << "Deleting tree..." << endl; };

    // copy assignment.
    tree& operator=(const tree<T>& other)
    {
      cout << "Beginning tree copy-assignment..." << endl;

      if (this == &other) return *this; // handle self-assignment

      // if the tree at the left-side of the assignment isn't empty,
      // it's cleared. Existing nodes are deleted.
      if (!is_empty())
      {
        cout << "Clearing out previous content at left-side..." << endl;
        root_.reset(nullptr);
      }

      // if the other tree isn't empty, it's copied from its root.
      if (!other.is_empty()) root_.reset(new node{*(other.root_)});

      return *this;
    }


#ifdef MOVE_SEMANTICS
    // move assignment
    tree& operator=(tree<T>&& other)
    {
      cout << "Beginning tree move-assignment..." << endl;

      // if the tree at the left-side of the assignment isn't empty,
      // it's cleared. Existing nodes are deleted.
      if (!is_empty())
      {
        cout << "Clearing out previous content at left-side..." << endl;
        root_.reset(nullptr);
      }
      // if the other tree isn't empty, its root is moved.
      if (!other.is_empty())
      {
        cout << "Moving root to the left-side." << endl;
        root_ = move(other.root_);
      }

      return *this;
    }
#endif // MOVE_SEMANTICS

    // pushes an element to the tree. At the root if empty
    // Or in the left branch if less or equal than the root value. In the right branch otherwise.
    friend tree& operator<<(tree& t, const T element)
    {
      // if the tree is empty, the root is just inaugurated.
      if (t.is_empty()) t.root_.reset(new node{element});
      // otherwise, the element is pushed from the root downward
      else (t.root_)->push_back(element);

      return t;
    }

    // true if the tree is not empty.
    bool is_empty() const
    { if (root_) return false; else return true; }

    // returns the tree elements in a vector. Order is undetermined.
    vector<T> to_vector() const
    {
      vector<T> v;

      if (!is_empty()) push_branch_to_vector(v, *root_);

      return v;
    }
  private:
    // Type node as a nested class of tree
    class node {
    public:
      // there's no default constructor
      // constructs a single node with the received value.
      node(const T value) : node_value_{value}, left_branch_{nullptr}, right_branch_{nullptr}
      { cout << "Creating leaf(" << value << ")." << endl; }

      // copy-constructor
      node(const node& other) : node_value_{other.node_value_},
        left_branch_{other.left_branch_ ? new node{*(other.left_branch_)} : nullptr},
        right_branch_{other.right_branch_ ? new node{*(other.right_branch_)} : nullptr}
      { cout << "Node(" << node_value_ << ") copy-constructed." << endl; }

      // destructor
      ~node() { cout << "Destroying node(" << node_value_ << ")." << endl; };

      // no copy-assignment
      node& operator=(const node& other) = delete;

      T value() const
      { return node_value_; }

      // pushes the received value to the left branch if less or equal than the node value. To the right branch otherwise.
      node& push_back(const T value) {
        if (value<=node_value_) {
          // if the left branch is not empty, the value is recursively pushed into.
          if (left_branch_) { left_branch_->push_back(value); }
          // if empty, it just gets inaugurated.
          else
          {
            cout << "At the left branch of node(" << node_value_ << "): ";
            left_branch_.reset(new node{value});
          }
        } else {
          // same comments about left_branch_ apply at the right one.
          if (right_branch_) { right_branch_->push_back(value); }
          else
          {
            cout << "At the right branch of node(" << node_value_ << "): ";
            right_branch_.reset(new node{value});
          }
        }

        return *this;
      }

      friend void push_branch_to_vector(vector<T>& v, const node& n)
      {
        v.push_back(n.value());
        if (n.left_branch_) push_branch_to_vector(v, *n.left_branch_);
        if (n.right_branch_) push_branch_to_vector(v, *n.right_branch_);
      }
    private:
      T node_value_;
      unique_ptr<node> left_branch_, right_branch_;
    };

    unique_ptr<node> root_;
  };
}

#endif // UNBALANCED_BINARY_TREE_H



// main.cpp
#include "unbalanced_binary_tree.h"
#include <string>
#include <vector>

using namespace unbalanced_binary_tree;
using namespace std;

tree<unsigned> make_word_size_tree(const tree<string>& word_tree) {
  cout << "Inside function make_word_size_tree()..." << endl;
  vector<string> words = word_tree.to_vector();

  cout << "tree<unsigned> t; // local variable declaration" << endl;
  tree<unsigned> t;

  for (auto word : words)
    t << word.size();

  cout << "Leaving function make_word_size_tree(). Local tree<unsigned> t about to lose visibility..." << endl;
  return t;
}

int main(int argc, char* argv[]) {
  cout << "tree<unsigned> integer_tree = { 1, 3, 5, 4 };" << endl;
  tree<unsigned> integer_tree = { 1, 3, 5, 4 };

  cout << endl << "tree<unsigned> it2 = integer_tree;" << endl;
  tree<unsigned> it2 = integer_tree;

  cout << endl << "tree<string> lincoln_tree;" << endl;
  tree<string> lincoln_tree;

  cout << "lincoln_tree << \"give\" << \"me\" << \"six\" << (...) << \"axe\";" << endl;
  lincoln_tree << "give" << "me" << "six" << "hours"
    << "to" << "chop" << "down" << "a" << "tree"
    << "and" << "I" << "will" << "spend" << "the" << "first" << "four"
    << "sharpening" << "the" << "axe";

  cout << endl << "integer_tree = make_word_size_tree(lincoln_tree);" << endl;
  integer_tree = make_word_size_tree(lincoln_tree);

  cout << endl << "it2 = integer_tree;" << endl;
  it2 = integer_tree;

  cout << endl << "Function main() finishing. Variables integer_tree, it2 and lincoln_tree about to lose visibility." << endl;
}
