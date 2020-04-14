import sys
import os
import io
import numpy as np
from multiprocessing import Queue, Process

from gensim.models.doc2vec import LabeledSentence

class ArrayReader(Process):
    def __init__(self, input_file, label_file, out_queue, batch_size, is_mlp=False, validate=False):
        super(ArrayReader, self).__init__()
        self.is_mlp = is_mlp
        self.validate = validate
        self.q = out_queue
        self.batch_size = batch_size
        self.input_file = input_file
        self.label_file = label_file

    def run(self):
        # x_file = np.load(self.input_file, mmap_mode='r')
        # y_file = np.load(self.label_file, mmap_mode='r')
        x_file = self.input_file
        y_file = self.label_file
        start_item = 0
        num_iter = 0
        while True:
            if start_item > y_file.shape[0]:
                # print('in new epoch for {}'.format(os.path.basename(self.input_file)))
                print('\nin new epoch for {}'.format(os.path.basename('X_data, y_train with validation data')))
                start_item = 0
            y_batch = y_file[start_item: start_item + self.batch_size]
            x_batch = x_file[start_item: start_item + self.batch_size]
            if self.is_mlp:
                x_batch = np.reshape(x_batch, (x_batch.shape[0], x_batch.shape[1] * x_batch.shape[2]))
            start_item += self.batch_size
            num_iter += 1
            try:
                self.q.put((x_batch, y_batch), block=True)
            except:
                return

class DocReader(Process):
    def __init__(self, level, level_type, preprocessed_files_prefix, out_queue, num_reader):
        super(DocReader, self).__init__()
        self.out_queue = out_queue
        self.num_reader = num_reader
        self.inference_docs_iterator = BatchWrapper(
            preprocessed_files_prefix,
            text_batch_size=None,
            buffer_size=READ_QUEUE_SIZE,
            level=level,
            level_type=level_type)

    def run(self):
        while True:
            for item in self.inference_docs_iterator:
                self.out_queue.put(item)
            for i in range(0, self.num_reader):
                self.out_queue.put(False, block=True, timeout=None)
            sys.exit()


class DocInferer(Process):
    def __init__(self, model, in_queue, out_queue):
        super(DocInferer, self).__init__()
        self.model = model
        self.in_queue = in_queue
        self.out_queue = out_queue

    def run(self):
        while True:
            doc_tuple = self.in_queue.get(block=True)
            if doc_tuple is False:
                self.out_queue.put(False)
                sys.exit()
            self.out_queue.put((doc_tuple[0], self.model.infer_vector(doc_tuple[1])))

###########################################################################################################################

class ExtendedPVDocumentBatchGenerator(Process):
    def __init__(self, filename_prefix, queue, start_file=0, offset=10000):
        super(ExtendedPVDocumentBatchGenerator, self).__init__()
        self.queue = queue
        self.offset = offset
        self.filename_prefix = filename_prefix
        self.files_loaded = start_file - offset

    def run(self):
        cur_file = None
        while True:
            try:
                if cur_file is None:
                    cur_file = io.BufferedReader(gzip.open(self.filename_prefix + str(self.files_loaded + self.offset) + '.gz'))
                    self.files_loaded += self.offset
                for line in cur_file:
                    self.queue.put(line)
                cur_file.close()
                cur_file = None
            except IOError:
                self.queue.put(False, block=True, timeout=None)
                info("All files are loaded - last file: {}".format(str(self.files_loaded + self.offset)))
                sys.exit()

class BatchWrapper(object):
    def __init__(self, training_preprocessed_files_prefix, buffer_size=10000, text_batch_size=10000, level=1, level_type=None):
        assert text_batch_size <= 10000 or text_batch_size is None
        self.level = level
        self.level_type = level_type[0]
        self.text_batch_size = text_batch_size
        self.q = Queue(maxsize=buffer_size)
        self.p = ExtendedPVDocumentBatchGenerator(training_preprocessed_files_prefix, queue=self.q,
                                                  start_file=0, offset=10000)
        self.p.start()
        self.cur_data = []

    def is_correct_type(self, doc_id):
        parts = doc_id.split("_")
        len_parts = len(parts)
        if len_parts == self.level:
            if len_parts == 1:
                return True
            if len_parts == self.level and (parts[1][0] == self.level_type or self.level_type is None):
                return True
        return False

    def return_sentences(self, line):
        line_array = tuple(line.split(" "))
        doc_id = line_array[0]
        if not self.is_correct_type(doc_id):
            return False
        line_array = line_array[1:]
        len_line_array = len(line_array)
        curr_batch_iter = 0
        # divide the document to batches according to the batch size
        sentences = []
        if self.text_batch_size is None:
            sentences.append((doc_id, line_array))
        else:
            while curr_batch_iter < len_line_array:
                sentences.append(LabeledSentence(words=line_array[curr_batch_iter: curr_batch_iter + self.text_batch_size], tags=[doc_id]))
                curr_batch_iter += self.text_batch_size
        return tuple(sentences)

    def __iter__(self):
        while True:
            item = self.q.get(block=True)
            if item is False:
                self.p.terminate()
                raise StopIteration()
            else:
                try:
                    sentences = self.return_sentences(item)
                except:
                    print(item)
                    raise StopIteration()
                if not sentences:
                    None
                else:
                    for sentence in sentences:
                        yield sentence