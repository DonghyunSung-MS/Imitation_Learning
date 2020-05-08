import utils.logger as logger

test = logger.Logger()
test.configure_output_file()

for i in range(1000):
    test.log_tabular("key1",i)
    test.log_tabular("key2",i*2)
    test.log_tabular("key3",i+2)
    test.log_tabular("key4",i*3)
    test.print_tabular()
    test.dump_tabular()
print(test.get_num_keys())
