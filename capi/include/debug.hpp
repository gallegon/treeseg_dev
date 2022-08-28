// #define TREESEG_DEBUG

#ifdef TREESEG_DEBUG
    #define DEBUG(...) do {__VA_ARGS__} while(0)
    #define DPRINT(xs) do {std::cout << xs << std::endl;} while(0)
    #define DVAR(...) __VA_ARGS__;
#else
    #define DEBUG(...) do {} while(0)
    #define DPRINT(xs) do {} while (0)
    #define DVAR(...) do {} while (0)
#endif // TREESEG_DEBUG