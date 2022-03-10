import sys

def main():
    system_major = sys.version_info.major
    system_minor = sys.version_info.minor

    if system_major != 3 or system_minor != 10:
        raise TypeError(
            "This project requires Python 3.10. Found: Python {}".format(
                sys.version))
    else:
        print(">>> Development environment passes all tests!")


if __name__ == '__main__':
    main()
