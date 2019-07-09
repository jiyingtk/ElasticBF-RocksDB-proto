//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "table/partitioned_filter_block.h"

#include <stdio.h>
#include <iostream>
#include <utility>

#include "monitoring/perf_context_imp.h"
#include "port/port.h"
#include "rocksdb/filter_policy.h"
#include "table/block.h"
#include "table/block_based_table_reader.h"
#include "util/coding.h"

namespace rocksdb {

PartitionedFilterBlockBuilder::PartitionedFilterBlockBuilder(
    const SliceTransform* prefix_extractor, bool whole_key_filtering,
    FilterBitsBuilder* filter_bits_builder, int index_block_restart_interval,
    PartitionedIndexBuilder* const p_index_builder,
    const uint32_t partition_size)
    : FullFilterBlockBuilder(prefix_extractor, whole_key_filtering,
                             filter_bits_builder),
      index_on_filter_block_builder_(index_block_restart_interval),
      p_index_builder_(p_index_builder),
      filters_in_partition_(0),
      num_added_(0) {
  filters_per_partition_ =
      filter_bits_builder_->CalculateNumEntry(partition_size);

  filter_nums = filter_bits_builder_->bits_per_keys_.size();
  filter_gc.resize(filter_nums);
  filters.resize(filter_nums);
  filter_index = region_index = 0;
  // for (int i = 0; i < filter_nums; i++) {
  //   filter_gc.push_back(std::unique_ptr<const char[]>(nullptr));

  //   filters.push_back({index_key, filter});
  // }
}

PartitionedFilterBlockBuilder::~PartitionedFilterBlockBuilder() {}

void PartitionedFilterBlockBuilder::MaybeCutAFilterBlock() {
  // Use == to send the request only once
  if (filters_in_partition_ == filters_per_partition_) {
    // Currently only index builder is in charge of cutting a partition. We keep
    // requesting until it is granted.
    p_index_builder_->RequestPartitionCut();
  }
  if (!p_index_builder_->ShouldCutFilterBlock()) {
    return;
  }
  for (int i = 0; i < filter_nums; i++) {
    filter_gc[i].push_back(std::unique_ptr<const char[]>(nullptr));
    Slice filter = filter_bits_builder_->Finish(&filter_gc[i].back(), i);
    std::string& index_key = p_index_builder_->GetPartitionKey();
    filters[i].push_back({index_key, filter});
  }
  filters_in_partition_ = 0;
  Reset();
}

void PartitionedFilterBlockBuilder::AddKey(const Slice& key) {
  MaybeCutAFilterBlock();
  filter_bits_builder_->AddKey(key);
  filters_in_partition_++;
  num_added_++;
}

Slice PartitionedFilterBlockBuilder::Finish(
    const BlockHandle& last_partition_block_handle, Status* status) {
  if (finishing_filters == true) {
    // Record the handle of the last written filter block in the index
    FilterEntry& last_entry = filters[filter_index].front();
    std::string handle_encoding;
    last_partition_block_handle.EncodeTo(&handle_encoding);

    // char prefix[] = "z00";
    // prefix[1] = '0' + (filter_index / 10);
    // prefix[2] = '0' + (filter_index % 10);
    char *new_key = new char[last_entry.key.size() + 8];
    EncodeFixed32R(new_key, filter_index);
    memcpy(new_key + 4, last_entry.key.c_str(), last_entry.key.size());
    EncodeFixed32R(new_key + last_entry.key.size() + 4, region_index++);

    index_on_filter_block_builder_.Add(Slice(new_key, last_entry.key.size() + 8), handle_encoding);
    delete new_key;
    filters[filter_index].pop_front();
  } else {
    MaybeCutAFilterBlock();
  }
  // If there is no filter partition left, then return the index on filter
  // partitions
  if (UNLIKELY(filters[filter_index].empty())) {
    filter_index++;
    region_index = 0;
    if (filter_index < filter_nums) {
      *status = Status::Incomplete();
      finishing_filters = true;
      return filters[filter_index].front().filter;
    }

    *status = Status::OK();
    if (finishing_filters) {
      return index_on_filter_block_builder_.Finish();
    } else {
      // This is the rare case where no key was added to the filter
      return Slice();
    }
  } else {
    // Return the next filter partition in line and set Incomplete() status to
    // indicate we expect more calls to Finish
    *status = Status::Incomplete();
    finishing_filters = true;
    return filters[filter_index].front().filter;
  }
}

PartitionedFilterBlockReader::PartitionedFilterBlockReader(
    const SliceTransform* prefix_extractor, bool _whole_key_filtering,
    BlockContents&& contents, FilterBitsReader* /*filter_bits_reader*/,
    Statistics* stats, const Comparator& comparator,
    const BlockBasedTable* table)
    : FilterBlockReader(contents.data.size(), stats, _whole_key_filtering),
      prefix_extractor_(prefix_extractor),
      comparator_(comparator),
      table_(table) {
  idx_on_fltr_blk_.reset(new Block(std::move(contents),
                                   kDisableGlobalSequenceNumber,
                                   0 /* read_amp_bytes_per_bit */, stats));
  InitRegionFilterInfo();
}

// #define ORIGINAL_VERSION

PartitionedFilterBlockReader::~PartitionedFilterBlockReader() {
  // TODO(myabandeh): if instead of filter object we store only the blocks in
  // block cache, then we don't have to manually earse them from block cache
  // here.
  // auto block_cache = table_->rep_->table_options.block_cache.get();
  auto block_cache = table_->rep_->table_options.metadata_cache.get();
  if (UNLIKELY(block_cache == nullptr)) {
    return;
  }
  char cache_key[BlockBasedTable::kMaxCacheKeyPrefixSize + kMaxVarint64Length];
  BlockIter biter;
  BlockHandle handle;
  idx_on_fltr_blk_->NewIterator(&comparator_, &biter, true);
  biter.SeekToFirst();
  for (; biter.Valid(); biter.Next()) {
    auto input = biter.value();
    auto s = handle.DecodeFrom(&input);
    assert(s.ok());
    if (!s.ok()) {
      continue;
    }
    auto key = BlockBasedTable::GetCacheKey(table_->rep_->cache_key_prefix,
                                            table_->rep_->cache_key_prefix_size,
                                            handle, cache_key);
    block_cache->Erase(key);
  }

#ifndef ORIGINAL_VERSION
  auto filter_info_cache = table_->rep_->table_options.filter_info_cache.get();
  for (int i = 0; i < region_nums; i++) {
    auto rkey = GetRegionCacheKey(cache_key, i);
    filter_info_cache->Erase(rkey);
  }
#endif
}

void DeleteRegionInfoEntry(const Slice& /*key*/, void* value) {
  auto entry = reinterpret_cast<RegionFilterInfo*>(value);
  delete entry;
}

void PartitionedFilterBlockReader::InitRegionFilterInfo() {
  BlockIter biter;
  idx_on_fltr_blk_->NewIterator(&comparator_, &biter, true);
  biter.SeekToFirst();
  int total_filter_nums = 0;
  for (; biter.Valid(); biter.Next()) {
    total_filter_nums++;
  }
  biter.SeekToLast();
  region_nums = DecodeFixed32R(biter.key().data() + (biter.key().size() - 4)) + 1;

#ifndef ORIGINAL_VERSION
  auto block_cache = table_->rep_->table_options.filter_info_cache.get();

  char* end = EncodeVarint64(cache_key_prefix, block_cache->NewId());
  cache_key_prefix_size = static_cast<size_t>(end - cache_key_prefix);


  static int output_count = 0;
  if (output_count++ < 5)
    fprintf(stderr, "table region_nums: %d\n", region_nums);

  int init_charge = 0;
  for (int i = 0; i < table_->rep_->table_options.init_filter_nums; i++)
    init_charge += table_->rep_->table_options.bits_per_key_per_filter[i];
  for (int i = 0; i < region_nums; i++) {
    RegionFilterInfo *info = new RegionFilterInfo;
    info->cur_filter_nums = table_->rep_->table_options.init_filter_nums;
    info->adjusted_filter_nums = table_->rep_->table_options.init_filter_nums;
    info->region_num = i;
    regionFilterInfos.push_back(info);

    char cache_key[BlockBasedTable::kMaxCacheKeyPrefixSize + kMaxVarint64Length];
    auto rkey = GetRegionCacheKey(cache_key, i);
    Status s = block_cache->Insert(rkey, info, init_charge, &DeleteRegionInfoEntry);
    if (!s.ok()) {
      fprintf(stderr, "insert region_num %d error!,%s\n", i, s.ToString().c_str());
    }
  }
#endif

}

#ifdef ORIGINAL_VERSION

bool PartitionedFilterBlockReader::KeyMayMatch(
    const Slice& key, uint64_t block_offset, const bool no_io,
    const Slice* const const_ikey_ptr, const int /*hash_id*/) {
  assert(const_ikey_ptr != nullptr);
  assert(block_offset == kNotValid);
  if (!whole_key_filtering_) {
    return true;
  }
  if (UNLIKELY(idx_on_fltr_blk_->size() == 0)) {
    return true;
  }

  for (int i = 0; i < table_->rep_->table_options.init_filter_nums; i++) {

    auto filter_handle = GetFilterPartitionHandle(*const_ikey_ptr, i);
    if (UNLIKELY(filter_handle.size() == 0)) {  // key is out of range
      return false;
    }
    bool cached = false;
    auto filter_partition = GetFilterPartition(nullptr /* prefetch_buffer */,
                                               &filter_handle, no_io, &cached);
    if (UNLIKELY(!filter_partition.value)) {
      continue;
    }
    auto res = filter_partition.value->KeyMayMatch(key, block_offset, no_io, nullptr, i);
    // if (cached) {
    // }
    if (LIKELY(filter_partition.IsSet())) {
      filter_partition.Release(table_->rep_->table_options.metadata_cache.get());
    } else {
      delete filter_partition.value;
    }
    if (res) 
      continue;
    else
      return false;
  }
  return true;
}

#endif

#ifndef ORIGINAL_VERSION

bool PartitionedFilterBlockReader::KeyMayMatch(
    const Slice& key, uint64_t block_offset, const bool no_io,
    const Slice* const const_ikey_ptr, const int /*hash_id*/) {
  assert(const_ikey_ptr != nullptr);
  assert(block_offset == kNotValid);
  if (!whole_key_filtering_) {
    return true;
  }
  if (UNLIKELY(idx_on_fltr_blk_->size() == 0)) {
    return true;
  }
  auto block_cache = table_->rep_->table_options.metadata_cache.get();


  auto region_info_entry = GetRegionInfoByKey(*const_ikey_ptr);

  if (UNLIKELY(region_info_entry.value == nullptr)) { // key is out of range
    return false;
  }

  RegionFilterInfo *region_filter_info = reinterpret_cast<RegionFilterInfo*>
      (region_info_entry.value);

  bool result = true;

  for (uint32_t i = 0; i < region_filter_info->adjusted_filter_nums; i++) {
    auto filter_handle = GetFilterPartitionHandle(*const_ikey_ptr, i);
    if (UNLIKELY(filter_handle.size() == 0)) {  // key is out of range
      result = false;
      break;
    }
    bool cached = false;
    
    auto filter_partition = GetFilterPartition(nullptr /* prefetch_buffer */,
                                               &filter_handle, no_io, &cached);
    if (UNLIKELY(!filter_partition.value)) {
      continue;
    }
    auto res = filter_partition.value->KeyMayMatch(key, block_offset, no_io, nullptr, i);
    // if (cached) {
    // }
    if (LIKELY(filter_partition.IsSet())) {
      filter_partition.Release(block_cache);
    } else {
      delete filter_partition.value;
    }
    result = res;
    if (!result)
      break;
  }

  for (uint32_t i = region_filter_info->adjusted_filter_nums; i < region_filter_info->cur_filter_nums; i++) {
    auto input = GetFilterPartitionHandle(*const_ikey_ptr, i);
    if (UNLIKELY(input.size() == 0)) {  // key is out of range
      continue;
    }
    BlockHandle filter_handle;
    auto s = filter_handle.DecodeFrom(&input);
    assert(s.ok());
    if (!s.ok()) {
      continue;
    }
    char cache_key[BlockBasedTable::kMaxCacheKeyPrefixSize + kMaxVarint64Length];
    auto fkey = BlockBasedTable::GetCacheKey(table_->rep_->cache_key_prefix,
                                            table_->rep_->cache_key_prefix_size,
                                            filter_handle, cache_key);
    block_cache->Erase(fkey);
  }
  region_filter_info->cur_filter_nums = region_filter_info->adjusted_filter_nums;

  if (LIKELY(region_info_entry.IsSet())) {
    region_info_entry.Release(table_->rep_->table_options.filter_info_cache.get());
  }
  else {
    fprintf(stderr, "error while release region_info_entry\n");
    assert(0);
  }
  return result;
}

#endif

bool PartitionedFilterBlockReader::PrefixMayMatch(
    const Slice& prefix, uint64_t block_offset, const bool no_io,
    const Slice* const const_ikey_ptr) {
#ifdef NDEBUG
  (void)block_offset;
#endif
  assert(const_ikey_ptr != nullptr);
  assert(block_offset == kNotValid);
  if (!prefix_extractor_) {
    return true;
  }
  if (UNLIKELY(idx_on_fltr_blk_->size() == 0)) {
    return true;
  }
  auto filter_handle = GetFilterPartitionHandle(*const_ikey_ptr);
  if (UNLIKELY(filter_handle.size() == 0)) {  // prefix is out of range
    return false;
  }
  bool cached = false;
  auto filter_partition = GetFilterPartition(nullptr /* prefetch_buffer */,
                                             &filter_handle, no_io, &cached);
  if (UNLIKELY(!filter_partition.value)) {
    return true;
  }
  auto res = filter_partition.value->PrefixMayMatch(prefix, kNotValid, no_io);
  if (cached) {
    return res;
  }
  if (LIKELY(filter_partition.IsSet())) {
    // filter_partition.Release(table_->rep_->table_options.block_cache.get());
    filter_partition.Release(table_->rep_->table_options.metadata_cache.get());
  } else {
    delete filter_partition.value;
  }
  return res;
}

Slice PartitionedFilterBlockReader::GetRegionCacheKey(char* cache_key, uint64_t region_num) {
  memcpy(cache_key, cache_key_prefix, cache_key_prefix_size);
  char* end =
    EncodeVarint64(cache_key + cache_key_prefix_size, region_num);
  return Slice(cache_key, static_cast<size_t>(end - cache_key));
}

BlockBasedTable::CachableEntry<RegionFilterInfo>
PartitionedFilterBlockReader::GetRegionInfoByKey(const Slice& entry) {
  BlockIter iter;
  idx_on_fltr_blk_->NewIterator(&comparator_, &iter, true);
  char *new_entry = new char[entry.size() + 4];

  EncodeFixed32R(new_entry, 0);
  memcpy(new_entry + 4, entry.data(), entry.size());

  iter.Seek(Slice(new_entry, entry.size() + 4));
  delete new_entry;
  if (UNLIKELY(!iter.Valid())) {
    return {nullptr, nullptr};
  }
  assert(iter.Valid());

  uint32_t region_num = DecodeFixed32R(iter.key().data() + (iter.key().size() - 4));

  // return {regionFilterInfos[region_num], nullptr};

  char cache_key[BlockBasedTable::kMaxCacheKeyPrefixSize + kMaxVarint64Length];
  auto rkey = GetRegionCacheKey(cache_key, region_num);

  Cache* block_cache = table_->rep_->table_options.filter_info_cache.get();
  auto cache_handle = block_cache->LookupRegion(rkey, table_->rep_->ioptions.statistics, true);
  if (cache_handle != nullptr) {
    RegionFilterInfo *region_filter_info = reinterpret_cast<RegionFilterInfo*>(
        block_cache->Value(cache_handle));
    assert(region_filter_info->region_num == region_num);
    return {region_filter_info, cache_handle};
  }
  else {
    fprintf(stderr, "get region info from cache error! region_nums: %u\n", region_num);

    if (regionFilterInfos[region_num] == nullptr) {
      fprintf(stderr, "regionFilterInfos null! region_nums: %u\n", region_num);
      exit(1);
    }
    Status s = block_cache->Insert(rkey, regionFilterInfos[region_num], regionFilterInfos[region_num]->cur_filter_nums * 4, &DeleteRegionInfoEntry);
    if (!s.ok()) {
      fprintf(stderr, "Insert regionFilterInfos false! region_nums: %u\n", region_num);
      exit(1);
    }
    cache_handle = block_cache->LookupRegion(rkey, table_->rep_->ioptions.statistics, true);

    if (cache_handle == nullptr) {
      fprintf(stderr, "LookupRegion still null! region_nums: %u\n", region_num);
      exit(1);
    }

    RegionFilterInfo *region_filter_info = reinterpret_cast<RegionFilterInfo*>(
            block_cache->Value(cache_handle));
    return {region_filter_info, cache_handle};

    // while (1) {
    //   int ii = 0;
    //   ii += 1;
    // }
    // assert(0);
    // return {nullptr, nullptr};
  }
}

Slice PartitionedFilterBlockReader::GetFilterPartitionHandle(
    const Slice& entry, const uint64_t filter_index) {
  BlockIter iter;
  idx_on_fltr_blk_->NewIterator(&comparator_, &iter, true);
  char *new_entry = new char[entry.size() + 4];
  // new_entry[0] = 'z';
  // new_entry[1] = '0';
  // new_entry[2] = '0';
  EncodeFixed32R(new_entry, filter_index);
  // EncodeFixed32R(new_entry + 4, 0);
  memcpy(new_entry + 4, entry.data(), entry.size());
  // iter.Seek(entry);
  iter.Seek(Slice(new_entry, entry.size() + 4));
  delete new_entry;
  if (UNLIKELY(!iter.Valid())) {
    return Slice();
  }
  assert(iter.Valid());
  Slice handle_value = iter.value();
  return handle_value;
}

BlockBasedTable::CachableEntry<FilterBlockReader>
PartitionedFilterBlockReader::GetFilterPartition(
    FilePrefetchBuffer* prefetch_buffer, Slice* handle_value, const bool no_io,
    bool* cached) {
  BlockHandle fltr_blk_handle;
  auto s = fltr_blk_handle.DecodeFrom(handle_value);
  assert(s.ok());
  const bool is_a_filter_partition = true;
  // auto block_cache = table_->rep_->table_options.block_cache.get();
  auto metadata_cache = table_->rep_->table_options.metadata_cache.get();
  if (LIKELY(metadata_cache != nullptr)) {
    if (filter_map_.size() != 0) {
      auto iter = filter_map_.find(fltr_blk_handle.offset());
      // This is a possible scenario since block cache might not have had space
      // for the partition
      if (iter != filter_map_.end()) {
        PERF_COUNTER_ADD(block_cache_hit_count, 1);
        RecordTick(statistics(), BLOCK_CACHE_FILTER_HIT);
        RecordTick(statistics(), BLOCK_CACHE_HIT);
        RecordTick(statistics(), BLOCK_CACHE_BYTES_READ,
                   metadata_cache->GetUsage(iter->second.cache_handle));
        *cached = true;
        return iter->second;
      }
    }
    return table_->GetFilter(/*prefetch_buffer*/ nullptr, fltr_blk_handle,
                             is_a_filter_partition, no_io,
                             /* get_context */ nullptr);
  } else {
    auto filter = table_->ReadFilter(prefetch_buffer, fltr_blk_handle,
                                     is_a_filter_partition);
    return {filter, nullptr};
  }
}

size_t PartitionedFilterBlockReader::ApproximateMemoryUsage() const {
  return idx_on_fltr_blk_->size();
}

// Release the cached entry and decrement its ref count.
void ReleaseFilterCachedEntry(void* arg, void* h) {
  Cache* cache = reinterpret_cast<Cache*>(arg);
  Cache::Handle* handle = reinterpret_cast<Cache::Handle*>(h);
  cache->Release(handle);
}

// TODO(myabandeh): merge this with the same function in IndexReader
void PartitionedFilterBlockReader::CacheDependencies(bool pin) {
  // Before read partitions, prefetch them to avoid lots of IOs
  auto rep = table_->rep_;
  BlockIter biter;
  BlockHandle handle;
  idx_on_fltr_blk_->NewIterator(&comparator_, &biter, true);
  // Index partitions are assumed to be consecuitive. Prefetch them all.
  // Read the first block offset
  biter.SeekToFirst();
  Slice input = biter.value();
  Status s = handle.DecodeFrom(&input);
  assert(s.ok());
  if (!s.ok()) {
    ROCKS_LOG_WARN(rep->ioptions.info_log,
                   "Could not read first index partition");
    return;
  }
  uint64_t prefetch_off = handle.offset();

  // Read the last block's offset
  biter.SeekToLast();
  input = biter.value();
  s = handle.DecodeFrom(&input);
  assert(s.ok());
  if (!s.ok()) {
    ROCKS_LOG_WARN(rep->ioptions.info_log,
                   "Could not read last index partition");
    return;
  }
  uint64_t last_off = handle.offset() + handle.size() + kBlockTrailerSize;
  uint64_t prefetch_len = last_off - prefetch_off;
  std::unique_ptr<FilePrefetchBuffer> prefetch_buffer;
  auto& file = table_->rep_->file;
  prefetch_buffer.reset(new FilePrefetchBuffer());
  s = prefetch_buffer->Prefetch(file.get(), prefetch_off,
    static_cast<size_t>(prefetch_len));

  // After prefetch, read the partitions one by one
  biter.SeekToFirst();
  // Cache* block_cache = rep->table_options.block_cache.get();
  Cache* metadata_cache = rep->table_options.metadata_cache.get();
  int count = 0;
  int max_load_filters = table_->rep_->table_options.init_filter_nums * region_nums;
  for (; biter.Valid() && count < max_load_filters; biter.Next(), count++) {
    input = biter.value();
    s = handle.DecodeFrom(&input);
    assert(s.ok());
    if (!s.ok()) {
      ROCKS_LOG_WARN(rep->ioptions.info_log, "Could not read index partition");
      continue;
    }

    const bool no_io = true;
    const bool is_a_filter_partition = true;
    auto filter = table_->GetFilter(prefetch_buffer.get(), handle,
                                    is_a_filter_partition, !no_io,
                                    /* get_context */ nullptr);
    if (LIKELY(filter.IsSet())) {
      if (pin) {
        filter_map_[handle.offset()] = std::move(filter);
        RegisterCleanup(&ReleaseFilterCachedEntry, metadata_cache,
                        filter.cache_handle);
      } else {
        metadata_cache->Release(filter.cache_handle);
      }
    } else {
      delete filter.value;
    }
  }
}

}  // namespace rocksdb
