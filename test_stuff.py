import apache_beam as beam


data_file = 'data/toy_data.csv'
with beam.Pipeline() as pipe:
    counters = (
        pipe
        | 'Read data' >> beam.io.ReadFromText(data_file, skip_header_lines=1)
      #  | 'Get max value' >> beam.CombinePerKey(max)
        | 'Get mean values' >> beam.CombineGlobally(lambda elem: max(elem or [None]))
        | 'Get means' >> beam.combiners.Mean.Globally()
        | 'Print values' >> beam.Map(print)
      #  | 'Divide'
        )

